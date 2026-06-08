from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from .model import TrainedArtifacts
from .rules import RuleHit, _has_executive_signal, apply_rules


RULE_HIGH_CONF = 0.90

MARGIN_UNSURE = 0.12


@dataclass
class Prediction:
    label: int
    confidence: float
    decided_by: str
    evidence_snippet: str
    evidence_source: str
    unsure: bool

    def to_dict(self) -> dict:
        return {
            "label": int(self.label),
            "confidence": float(self.confidence),
            "decided_by": self.decided_by,
            "evidence_snippet": self.evidence_snippet,
            "evidence_source": self.evidence_source,
            "unsure": bool(self.unsure),
        }


def predict_one(title: str, description: str,
                artifacts: TrainedArtifacts) -> Prediction:
    return predict_batch([{"title": title, "description": description}],
                         artifacts)[0]


def predict_batch(rows: list[dict],
                  artifacts: TrainedArtifacts) -> list[Prediction]:
   
    if not rows:
        return []

    rule_hits: list[Optional[RuleHit]] = [
        apply_rules(r.get("title") or "", r.get("description") or "")
        for r in rows
    ]

    proba = artifacts.pipeline.predict_proba(rows)
    classes = artifacts.classes_

    out: list[Prediction] = []
    for i, hit in enumerate(rule_hits):
        p = proba[i]
        order = np.argsort(p)[::-1]
        top1, top2 = order[0], order[1]
        model_label = int(classes[top1])
        model_conf = float(p[top1])
        margin = float(p[top1] - p[top2])

        if hit is not None and hit.confidence >= RULE_HIGH_CONF:
            label = hit.label
            conf = hit.confidence
            decided_by = hit.decided_by
            snippet = hit.evidence_snippet
            source = hit.evidence_source
            unsure = abs(label - model_label) >= 2
        elif hit is not None:
            if hit.label == model_label:
                label = model_label
                conf = min(1.0, model_conf + 0.10)
                decided_by = f"{hit.decided_by}+model"
                snippet = hit.evidence_snippet
                source = hit.evidence_source
                unsure = (conf < artifacts.threshold and margin < MARGIN_UNSURE)
            else:
                label = model_label
                conf = model_conf
                decided_by = "model"
                snippet = hit.evidence_snippet
                source = hit.evidence_source if hit.evidence_snippet else "none"
                unsure = True
        else:
            label = model_label
            conf = model_conf
            decided_by = "model"
            snippet = ""
            source = "none"
            unsure = (model_conf < artifacts.threshold) or (margin < MARGIN_UNSURE)

        # Executive-floor guardrail
        title = rows[i].get("title") or ""
        if label == 1 and _has_executive_signal(title) and (
            hit is None or hit.confidence < RULE_HIGH_CONF
        ):
            label = 4
            decided_by = decided_by + "+executive_guardrail"
            if not snippet:
                snippet = title
                source = "title"
            unsure = True

        out.append(Prediction(
            label=label,
            confidence=conf,
            decided_by=decided_by,
            evidence_snippet=snippet,
            evidence_source=source,
            unsure=unsure,
        ))
    return out
