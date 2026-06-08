"""Train the YoE classifier.

- Loads gold + silver, joins descriptions from the jobs CSV.
- Runs 5-fold stratified CV on gold (silver always in train) for each
  candidate estimator. Picks the winner by macro-F1.
- Tunes the model-confidence `threshold` for `unsure` flagging on the same
  CV.
- Refits the winner on gold + silver with sample weights.
- Persists pipeline + meta to ./artifacts/.

Usage:
    python -m yoe_classifier.cli.train [--data-dir DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error
from sklearn.model_selection import StratifiedKFold

from ..data_io import LabeledRow, load_jobs_csv, load_labels, rows_to_dicts
from ..model import (CANDIDATES, TrainedArtifacts, fit_with_weights,
                     make_sample_weights)
from ..predict import MARGIN_UNSURE, RULE_HIGH_CONF
from ..rules import _has_executive_signal, apply_rules


# Candidate `unsure` thresholds to sweep.
THRESHOLD_GRID = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def _hybrid_predict(test_rows: list[LabeledRow],
                    proba: np.ndarray,
                    classes: list[int],
                    threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same hybrid combiner as `predict.predict_batch`.

    Returns (predicted_labels, flagged) -- both length n.
    """
    n = len(test_rows)
    pred = np.zeros(n, dtype=np.int64)
    flagged = np.zeros(n, dtype=bool)
    classes = list(classes)
    col = {int(c): j for j, c in enumerate(classes)}
    for i, row in enumerate(test_rows):
        p = proba[i]
        order = np.argsort(p)[::-1]
        top1, top2 = order[0], order[1]
        model_label = int(classes[top1])
        model_conf = float(p[top1])
        margin = float(p[top1] - p[top2])
        hit = apply_rules(row.title, row.description)
        if hit is not None and hit.confidence >= RULE_HIGH_CONF:
            pred[i] = hit.label
            flagged[i] = abs(hit.label - model_label) >= 2
        elif hit is not None and hit.label == model_label:
            pred[i] = model_label
            flagged[i] = (model_conf + 0.10 < threshold and margin < MARGIN_UNSURE)
        elif hit is not None:
            pred[i] = model_label
            flagged[i] = True
        else:
            pred[i] = model_label
            flagged[i] = (model_conf < threshold) or (margin < MARGIN_UNSURE)

        # Executive-floor guardrail -- mirrors predict.predict_batch so CV
        # evaluation reflects production behavior.
        if pred[i] == 1 and _has_executive_signal(row.title or "") and (
            hit is None or hit.confidence < RULE_HIGH_CONF
        ):
            pred[i] = 4
            flagged[i] = True
    return pred, flagged


def _cv_eval(estimator_factory, gold: list[LabeledRow], silver: list[LabeledRow],
             n_splits: int = 5, seed: int = 0) -> dict:
    """5-fold CV on gold (silver always in train).

    Returns OOF model probabilities AND the test-row LabeledRow list aligned
    to gold order, so we can compute hybrid (rules+model) metrics later.
    """
    y_gold = np.array([r.label for r in gold], dtype=np.int64)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n = len(gold)
    oof_proba = np.zeros((n, 4), dtype=np.float64)
    classes_full = [1, 2, 3, 4]

    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(n), y_gold)):
        tr_rows = [gold[i] for i in tr_idx] + silver
        te_rows = [gold[i] for i in te_idx]
        tr_y = np.array([r.label for r in tr_rows], dtype=np.int64)
        tr_w = make_sample_weights(
            [r.source for r in tr_rows],
            [r.unsure for r in tr_rows],
        )
        est = estimator_factory()
        fit_with_weights(est, rows_to_dicts(tr_rows), tr_y, tr_w)
        proba = est.predict_proba(rows_to_dicts(te_rows))
        col = {int(c): j for j, c in enumerate(est.classes_)}
        for k, lab in enumerate(classes_full):
            if lab in col:
                oof_proba[te_idx, k] = proba[:, col[lab]]

    # Model-only OOF predictions (for ranking estimators)
    model_pred = np.argmax(oof_proba, axis=1) + 1

    return {
        "rows":        gold,            # aligned to y_true
        "y_true":      y_gold,
        "model_pred":  model_pred,
        "y_proba":     oof_proba,
        "classes":     classes_full,
    }


def _summarize(name: str, cv: dict, threshold: float) -> dict:
    """Compute HYBRID (rules + model) metrics at a given threshold."""
    y_true = cv["y_true"]
    rows   = cv["rows"]
    proba  = cv["y_proba"]
    model_pred = cv["model_pred"]
    hyb_pred, flagged = _hybrid_predict(rows, proba, cv["classes"], threshold)
    kept = ~flagged
    return {
        "estimator": name,
        "threshold": threshold,
        "n": int(len(y_true)),
        "model_only_accuracy": float(accuracy_score(y_true, model_pred)),
        "model_only_macro_f1": float(
            f1_score(y_true, model_pred, average="macro", zero_division=0)),
        "hybrid_accuracy": float(accuracy_score(y_true, hyb_pred)),
        "hybrid_macro_f1": float(
            f1_score(y_true, hyb_pred, average="macro", zero_division=0)),
        "hybrid_mae":      float(mean_absolute_error(y_true, hyb_pred)),
        "coverage": float(kept.mean()),
        "kept_accuracy": (
            float(accuracy_score(y_true[kept], hyb_pred[kept])) if kept.any() else 0.0
        ),
        "flagged_pct": float(flagged.mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Train YoE classifier")
    ap.add_argument("--data-dir", default="yoe_classifier/data")
    ap.add_argument("--out-dir",  default="yoe_classifier/artifacts")
    ap.add_argument("--gold",     default="gold100.json")
    ap.add_argument("--silver",   default="silver400.json")
    ap.add_argument("--jobs",     default="jobs.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)
    print(f"[train] loading data from {data_dir}/")
    jobs = load_jobs_csv(data_dir / args.jobs)
    gold   = load_labels(data_dir / args.gold,   "gold",   jobs)
    silver = load_labels(data_dir / args.silver, "silver", jobs)
    print(f"[train] gold={len(gold)}  silver={len(silver)}  jobs_in_csv={len(jobs)}")

    # 1) CV-pick the best estimator
    print(f"\n[train] 5-fold CV on gold (silver in train fold) -- comparing candidates")
    best = None
    cv_summaries: dict[str, dict] = {}
    cv_cache: dict[str, dict] = {}
    for name, factory in CANDIDATES.items():
        try:
            probe = factory()
        except Exception as e:
            print(f"  - {name}: skipped ({e})")
            continue
        if probe is None:
            print(f"  - {name}: skipped (not installed)")
            continue
        t0 = time.time()
        try:
            cv = _cv_eval(factory, gold, silver)
        except Exception as e:
            print(f"  - {name}: ERROR {e}")
            continue
        elapsed = time.time() - t0
        # Default threshold = 0.55 just for ranking
        summary = _summarize(name, cv, threshold=0.55)
        cv_summaries[name] = summary
        cv_cache[name] = cv
        print(f"  - {name:>14}: hybrid_macroF1={summary['hybrid_macro_f1']:.3f}  "
              f"hybrid_acc={summary['hybrid_accuracy']:.3f}  "
              f"mae={summary['hybrid_mae']:.3f}  "
              f"(model-only acc={summary['model_only_accuracy']:.3f})  "
              f"({elapsed:.1f}s)")
        if best is None or summary["hybrid_macro_f1"] > best["hybrid_macro_f1"]:
            best = summary

    if best is None:
        print("[train] FATAL: no candidate succeeded")
        return 1

    print(f"\n[train] winner: {best['estimator']}  "
          f"(hybrid_macroF1={best['hybrid_macro_f1']:.3f})")
    winner_name = best["estimator"]
    winner_cv = cv_cache[winner_name]

    # 2) Tune threshold for the unsure flag against HYBRID kept-accuracy.
    print(f"\n[train] tuning unsure threshold for winner")
    threshold_summaries = []
    for t in THRESHOLD_GRID:
        s = _summarize(winner_name, winner_cv, threshold=t)
        threshold_summaries.append(s)
        print(f"  - thr={t:.2f}: kept_acc={s['kept_accuracy']:.3f}  "
              f"coverage={s['coverage']:.1%}  flagged={s['flagged_pct']:.1%}")
    # Pick the smallest threshold whose kept_accuracy >= 0.95 AND
    # flagged <= 0.30. If none, drop to >=0.90 / <=0.40. If still none, fall
    # back to the threshold maximizing kept_acc * sqrt(coverage).
    eligible = [s for s in threshold_summaries
                if s["kept_accuracy"] >= 0.95 and s["flagged_pct"] <= 0.30]
    if not eligible:
        eligible = [s for s in threshold_summaries
                    if s["kept_accuracy"] >= 0.90 and s["flagged_pct"] <= 0.40]
    if eligible:
        chosen = min(eligible, key=lambda s: s["threshold"])
    else:
        chosen = max(threshold_summaries,
                     key=lambda s: s["kept_accuracy"] * (s["coverage"] ** 0.5))
    print(f"[train] picked threshold={chosen['threshold']:.2f}  "
          f"(kept_acc={chosen['kept_accuracy']:.3f}, coverage={chosen['coverage']:.1%})")

    # 3) Refit winner on ALL data (gold + silver) with sample weights
    print(f"\n[train] refitting winner on gold+silver")
    all_rows = gold + silver
    y_all = np.array([r.label for r in all_rows], dtype=np.int64)
    w_all = make_sample_weights(
        [r.source for r in all_rows],
        [r.unsure for r in all_rows],
    )
    final = CANDIDATES[winner_name]()
    fit_with_weights(final, rows_to_dicts(all_rows), y_all, w_all)

    artifacts = TrainedArtifacts(
        pipeline=final,
        estimator_name=winner_name,
        threshold=float(chosen["threshold"]),
        classes_=np.asarray(final.classes_, dtype=np.int64),
        cv_summary={
            "winner": best,
            "all_candidates": cv_summaries,
            "threshold_grid": threshold_summaries,
            "chosen_threshold": chosen,
        },
    )
    artifacts.save(out_dir)
    print(f"\n[train] saved artifacts to {out_dir}/")
    print(f"        - model.joblib")
    print(f"        - meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
