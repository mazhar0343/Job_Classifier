"""Cross-validation report for the YoE classifier.

Re-runs 5-fold CV on gold (silver always in train) using the SAME estimator
that `cli/train.py` picked, then prints:

  - hybrid macro-F1 / accuracy / MAE / per-class precision-recall
  - confusion matrix
  - rules-only precision and coverage
  - coverage-vs-accuracy table swept over THRESHOLD_GRID

Usage:
    python -m yoe_classifier.cli.evaluate [--data-dir DIR] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score,
                             mean_absolute_error)

from ..data_io import load_jobs_csv, load_labels
from ..model import CANDIDATES, TrainedArtifacts
from ..rules import apply_rules
from .train import THRESHOLD_GRID, _cv_eval, _hybrid_predict, _summarize


def _print_confusion(y_true, y_pred, labels=(1, 2, 3, 4)) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(labels))
    print("    pred ->  " + "  ".join(f"{l:>5d}" for l in labels) + "    sum")
    for i, l in enumerate(labels):
        row = "  ".join(f"{v:>5d}" for v in cm[i])
        print(f"    true {l}:  {row}    {cm[i].sum():>5d}")
    print("    sum   :  " + "  ".join(f"{v:>5d}" for v in cm.sum(axis=0))
          + f"    {cm.sum():>5d}")


def _rules_only_report(rows) -> None:
    """Precision/coverage of rules ALONE (no ML)."""
    total = len(rows)
    by_conf = {"high (>=0.9)": [0, 0], "low (<0.9)": [0, 0]}
    by_decided = {}
    fired_high = 0
    for r in rows:
        hit = apply_rules(r.title, r.description)
        if hit is None:
            continue
        bucket = "high (>=0.9)" if hit.confidence >= 0.9 else "low (<0.9)"
        by_conf[bucket][1] += 1
        by_decided.setdefault(hit.decided_by, [0, 0])
        by_decided[hit.decided_by][1] += 1
        if hit.confidence >= 0.9:
            fired_high += 1
        if hit.label == r.label:
            by_conf[bucket][0] += 1
            by_decided[hit.decided_by][0] += 1
    print(f"  total rows: {total}")
    print(f"  rules fired (high conf): {fired_high}/{total} = {fired_high/total:.1%}")
    print(f"  precision by confidence bucket:")
    for k, (c, t) in by_conf.items():
        if t:
            print(f"    {k:>14}: {c}/{t} = {c/t:.1%}")
    print(f"  precision by decided_by:")
    for k, (c, t) in sorted(by_decided.items(), key=lambda x: -x[1][1]):
        print(f"    {k:>20}: {c}/{t} = {c/t:.1%}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate YoE classifier")
    ap.add_argument("--data-dir", default="yoe_classifier/data")
    ap.add_argument("--out-dir",  default="yoe_classifier/artifacts")
    ap.add_argument("--gold",     default="gold100.json")
    ap.add_argument("--silver",   default="silver400.json")
    ap.add_argument("--jobs",     default="jobs.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    print(f"[evaluate] loading data from {data_dir}/")
    jobs = load_jobs_csv(data_dir / args.jobs)
    gold   = load_labels(data_dir / args.gold,   "gold",   jobs)
    silver = load_labels(data_dir / args.silver, "silver", jobs)
    print(f"[evaluate] gold={len(gold)}  silver={len(silver)}")

    # Discover the winner from artifacts/meta.json
    artifacts = TrainedArtifacts.load(out_dir)
    name = artifacts.estimator_name
    threshold = artifacts.threshold
    print(f"[evaluate] using estimator={name!r} threshold={threshold:.2f}")

    # Re-run CV with the picked estimator (silver always in train)
    print(f"\n[evaluate] running 5-fold CV on gold (silver in train)")
    factory = CANDIDATES[name]
    cv = _cv_eval(factory, gold, silver)

    # Hybrid metrics at the chosen threshold
    summary = _summarize(name, cv, threshold=threshold)

    print(f"\n=== HYBRID (rules + model) at threshold={threshold:.2f} ===")
    print(f"  accuracy:            {summary['hybrid_accuracy']:.3f}")
    print(f"  macro-F1:            {summary['hybrid_macro_f1']:.3f}")
    print(f"  ordinal MAE:         {summary['hybrid_mae']:.3f}")
    print(f"  coverage (kept):     {summary['coverage']:.1%}")
    print(f"  flagged (unsure):    {summary['flagged_pct']:.1%}")
    print(f"  kept-only accuracy:  {summary['kept_accuracy']:.3f}")
    print(f"  model-only accuracy: {summary['model_only_accuracy']:.3f}  "
          f"(macro-F1 {summary['model_only_macro_f1']:.3f})")

    # Per-class report + confusion matrix on the HYBRID predictions
    rows = cv["rows"]; y_true = cv["y_true"]; proba = cv["y_proba"]
    hyb_pred, flagged = _hybrid_predict(rows, proba, cv["classes"], threshold)
    print(f"\n=== Per-class report (hybrid, all rows) ===")
    print(classification_report(
        y_true, hyb_pred, labels=[1, 2, 3, 4],
        target_names=["1: 0-2y (entry)", "2: 3-5y (early)",
                      "3: 6-15y (mid)",  "4: 15+y (senior)"],
        zero_division=0, digits=3,
    ))
    print("=== Confusion matrix (hybrid, all rows) ===")
    _print_confusion(y_true, hyb_pred)

    # Per-class on the KEPT (non-flagged) rows
    kept = ~flagged
    if kept.any():
        print(f"\n=== Confusion matrix (hybrid, kept rows only, n={kept.sum()}) ===")
        _print_confusion(y_true[kept], hyb_pred[kept])

    # Rules-only report on gold
    print("\n=== Rules-only precision (gold) ===")
    _rules_only_report(gold)
    print("\n=== Rules-only precision (silver) ===")
    _rules_only_report(silver)

    # Coverage-vs-accuracy curve
    print("\n=== Coverage vs accuracy sweep (hybrid, on gold OOF) ===")
    print(f"    {'thr':>6}  {'kept_acc':>9}  {'coverage':>9}  {'flagged':>8}  {'macroF1':>8}  {'mae':>6}")
    for t in THRESHOLD_GRID:
        s = _summarize(name, cv, threshold=t)
        print(f"    {t:>6.2f}  {s['kept_accuracy']:>9.3f}  "
              f"{s['coverage']:>9.1%}  {s['flagged_pct']:>8.1%}  "
              f"{s['hybrid_macro_f1']:>8.3f}  {s['hybrid_mae']:>6.3f}")

    # Persist evaluation report
    report = {
        "estimator": name,
        "threshold": threshold,
        "summary": summary,
        "confusion_matrix": confusion_matrix(y_true, hyb_pred,
                                             labels=[1, 2, 3, 4]).tolist(),
        "label_distribution_true": dict(Counter(int(x) for x in y_true)),
        "label_distribution_pred": dict(Counter(int(x) for x in hyb_pred)),
        "threshold_sweep": [_summarize(name, cv, threshold=t)
                            for t in THRESHOLD_GRID],
    }
    (out_dir / "evaluation.json").write_text(json.dumps(report, indent=2, default=int))
    print(f"\n[evaluate] full report written to {out_dir/'evaluation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
