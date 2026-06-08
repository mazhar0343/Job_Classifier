from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from .features import build_pipeline

GOLD_WEIGHT = 3.0
SILVER_WEIGHT = 1.0
SILVER_UNSURE_WEIGHT = 0.3


def make_sample_weights(sources: list[str], unsure: list[bool]) -> np.ndarray:
    out = np.empty(len(sources), dtype=np.float64)
    for i, (s, u) in enumerate(zip(sources, unsure)):
        if s == "gold":
            out[i] = GOLD_WEIGHT
        elif s == "silver" and u:
            out[i] = SILVER_UNSURE_WEIGHT
        else:
            out[i] = SILVER_WEIGHT
    return out


def _logreg(C: float = 1.0) -> Pipeline:
    base = LogisticRegression(
        solver="lbfgs",
        max_iter=2000,
        C=C,
        class_weight="balanced",
        random_state=0,
    )
    return Pipeline([
        ("features", build_pipeline()),
        ("clf", CalibratedClassifierCV(base, method="sigmoid", cv=3)),
    ])


def _linear_svc(C: float = 1.0) -> Pipeline:
    base = LinearSVC(C=C, class_weight="balanced", max_iter=4000, random_state=0)
    return Pipeline([
        ("features", build_pipeline()),
        ("clf", CalibratedClassifierCV(base, method="sigmoid", cv=3)),
    ])


def _lightgbm() -> Optional[Pipeline]:
    try:
        from lightgbm import LGBMClassifier
    except Exception:
        return None
    base = LGBMClassifier(
        n_estimators=400,
        num_leaves=31,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=0,
        verbose=-1,
    )
    return Pipeline([
        ("features", build_pipeline()),
        ("clf", CalibratedClassifierCV(base, method="sigmoid", cv=3)),
    ])


CANDIDATES: dict[str, callable] = {
    "logreg_C0.5": lambda: _logreg(C=0.5),
    "logreg_C1.0": lambda: _logreg(C=1.0),
    "logreg_C2.0": lambda: _logreg(C=2.0),
    "linsvc_C0.5": lambda: _linear_svc(C=0.5),
    "linsvc_C1.0": lambda: _linear_svc(C=1.0),
    "lightgbm":    _lightgbm,
}


@dataclass
class TrainedArtifacts:
    pipeline: Pipeline
    estimator_name: str
    threshold: float
    classes_: np.ndarray
    cv_summary: dict = field(default_factory=dict)

    def save(self, dirpath: str | Path) -> None:
        d = Path(dirpath)
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, d / "model.joblib")
        meta = {
            "estimator_name": self.estimator_name,
            "threshold": float(self.threshold),
            "classes": [int(c) for c in self.classes_],
            "cv_summary": self.cv_summary,
        }
        (d / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, dirpath: str | Path) -> "TrainedArtifacts":
        d = Path(dirpath)
        pipeline = joblib.load(d / "model.joblib")
        meta = json.loads((d / "meta.json").read_text())
        return cls(
            pipeline=pipeline,
            estimator_name=meta["estimator_name"],
            threshold=float(meta["threshold"]),
            classes_=np.asarray(meta["classes"], dtype=np.int64),
            cv_summary=meta.get("cv_summary", {}),
        )



def predict_proba(pipeline: Pipeline, rows: list[dict]) -> np.ndarray:
    """Run `predict_proba` on a list of row dicts."""
    return pipeline.predict_proba(rows)


def fit_with_weights(estimator: Pipeline, rows: list[dict],
                     y: np.ndarray, sample_weight: np.ndarray) -> Pipeline:
    """Fit a Pipeline that ends in CalibratedClassifierCV with sample weights.

    `CalibratedClassifierCV.fit` accepts `sample_weight` directly when the
    underlying base estimator does.
    """
    estimator.fit(rows, y, clf__sample_weight=sample_weight)
    return estimator
