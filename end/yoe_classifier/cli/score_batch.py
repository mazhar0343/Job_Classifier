"""Batch-score a CSV of jobs.

Input CSV: must contain `UniqueJobAdID`, `OrigJobTitle`, `JobDescription`
(matches the format of `Candogram-5000-random-jobs.csv`).

Output CSV columns mirror the gold/silver labeling schema:
    job_id, title, label, confidence, decided_by,
    evidence_snippet, evidence_source, unsure

Vectorized: rules run row-wise, the model gets a single `predict_proba`
batch call.

Usage:
    python -m yoe_classifier.cli.score_batch \
        --input data/jobs.csv --output predictions.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

from ..data_io import load_jobs_csv
from ..model import TrainedArtifacts
from ..predict import predict_batch

csv.field_size_limit(sys.maxsize)


def _iter_input(path: str | Path):
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            yield row


def main() -> int:
    ap = argparse.ArgumentParser(description="Batch-score jobs CSV")
    ap.add_argument("--input",  required=True, help="path to jobs CSV")
    ap.add_argument("--output", required=True, help="path to predictions CSV")
    ap.add_argument("--artifacts-dir", default="yoe_classifier/artifacts")
    ap.add_argument("--batch-size", type=int, default=2000)
    ap.add_argument("--only-flagged", action="store_true",
                    help="write only rows where unsure=True (for relabeling)")
    args = ap.parse_args()

    print(f"[score] loading model from {args.artifacts_dir}/")
    artifacts = TrainedArtifacts.load(args.artifacts_dir)
    print(f"[score] estimator={artifacts.estimator_name}  "
          f"threshold={artifacts.threshold:.2f}")

    in_path  = Path(args.input)
    out_path = Path(args.output)
    print(f"[score] reading {in_path} -> writing {out_path}")

    n_total = 0
    n_flagged = 0
    t0 = time.time()

    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["job_id", "title", "label", "confidence",
                         "decided_by", "evidence_snippet",
                         "evidence_source", "unsure"])

        batch_meta: list[tuple[str, str]] = []   # (job_id, title)
        batch_rows: list[dict] = []

        def flush() -> None:
            nonlocal n_total, n_flagged
            if not batch_rows:
                return
            preds = predict_batch(batch_rows, artifacts)
            for (jid, title), p in zip(batch_meta, preds):
                n_total += 1
                if p.unsure:
                    n_flagged += 1
                if args.only_flagged and not p.unsure:
                    continue
                writer.writerow([
                    jid,
                    title,
                    p.label,
                    f"{p.confidence:.3f}",
                    p.decided_by,
                    p.evidence_snippet,
                    p.evidence_source,
                    "true" if p.unsure else "false",
                ])
            batch_meta.clear()
            batch_rows.clear()

        for row in _iter_input(in_path):
            jid = row.get("UniqueJobAdID") or row.get("job_id") or ""
            title = row.get("OrigJobTitle") or row.get("title") or ""
            desc = row.get("JobDescription") or row.get("description") or ""
            batch_meta.append((jid, title))
            batch_rows.append({"title": title, "description": desc})
            if len(batch_rows) >= args.batch_size:
                flush()
                elapsed = time.time() - t0
                rate = n_total / max(elapsed, 1e-6)
                print(f"  scored {n_total:>6,d} rows  "
                      f"({rate:>5.0f}/s, flagged {n_flagged}/{n_total} = "
                      f"{n_flagged/max(n_total,1):.1%})")
        flush()

    elapsed = time.time() - t0
    print(f"\n[score] done.")
    print(f"  total rows:   {n_total:,}")
    print(f"  flagged:      {n_flagged:,} ({n_flagged/max(n_total,1):.1%})")
    print(f"  elapsed:      {elapsed:.1f}s ({n_total/max(elapsed,1e-6):.0f} rows/s)")
    print(f"  written to:   {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
