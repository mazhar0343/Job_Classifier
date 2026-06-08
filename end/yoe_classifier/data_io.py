"""Data loaders shared by the CLIs."""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# JD bodies can be huge; lift the CSV limit once at import time.
csv.field_size_limit(sys.maxsize)


@dataclass
class LabeledRow:
    job_id: str
    title: str
    description: str
    label: int
    source: str          # "gold" or "silver"
    unsure: bool = False
    decided_by: str = ""
    evidence_snippet: str = ""
    evidence_source: str = ""


def load_jobs_csv(path: str | Path) -> dict[str, tuple[str, str]]:
    """Return {job_id: (title, description)}."""
    out: dict[str, tuple[str, str]] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[row["UniqueJobAdID"]] = (
                row.get("OrigJobTitle") or "",
                row.get("JobDescription") or "",
            )
    return out


def load_labels(path: str | Path, source: str,
                jobs: dict[str, tuple[str, str]]) -> list[LabeledRow]:
    """Load gold/silver JSON, joining titles+descriptions from `jobs`."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    out: list[LabeledRow] = []
    skipped = 0
    for d in data:
        jid = str(d["job_id"])
        if jid not in jobs:
            skipped += 1
            continue
        title, desc = jobs[jid]
        out.append(LabeledRow(
            job_id=jid,
            title=title or d.get("title") or "",
            description=desc,
            label=int(d["label"]),
            source=source,
            unsure=bool(d.get("unsure", False)),
            decided_by=d.get("decided_by") or "",
            evidence_snippet=d.get("evidence_snippet") or "",
            evidence_source=d.get("evidence_source") or "",
        ))
    if skipped:
        print(f"[data_io] warning: {skipped} {source} rows had no matching job in CSV",
              file=sys.stderr)
    return out


def rows_to_dicts(rows: list[LabeledRow]) -> list[dict]:
    """Convert to the {title, description} dict form the pipeline expects."""
    return [{"title": r.title, "description": r.description} for r in rows]
