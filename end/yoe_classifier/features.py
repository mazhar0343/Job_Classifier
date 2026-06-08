from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler

from .rules import extract_yoe_features


_JD_TRUNC = 8000


_NUMERIC_FEATURES = [
    "max_yoe_floor",
    "min_yoe_floor",
    "max_yoe_ceil",
    "n_yoe_matches",
    "has_yoe_match",
    "has_negation",
    "has_edu_substitute",
    "edu_alt_yrs",
    "has_mgmt_scope",
    "mgmt_n",
    "title_seniority_label",
    "has_title_seniority",
    "jd_char_len",
    "jd_word_len",
    "title_word_len",
    "bullet_count",
]



def _get_title(rows: Iterable[dict]) -> list[str]:
    return [(r.get("title") or "") for r in rows]


def _get_description(rows: Iterable[dict]) -> list[str]:
    return [(r.get("description") or "")[:_JD_TRUNC] for r in rows]


def _get_numeric(rows: Iterable[dict]) -> sparse.csr_matrix:
    rows = list(rows)
    out = np.zeros((len(rows), len(_NUMERIC_FEATURES)), dtype=np.float64)
    for i, r in enumerate(rows):
        feats = extract_yoe_features(r.get("title") or "", r.get("description") or "")
        for j, name in enumerate(_NUMERIC_FEATURES):
            out[i, j] = feats[name]
    out[:, _NUMERIC_FEATURES.index("jd_char_len")] = np.log1p(
        out[:, _NUMERIC_FEATURES.index("jd_char_len")]
    )
    out[:, _NUMERIC_FEATURES.index("jd_word_len")] = np.log1p(
        out[:, _NUMERIC_FEATURES.index("jd_word_len")]
    )
    return sparse.csr_matrix(out)



def build_pipeline() -> FeatureUnion:
    title_word = Pipeline([
        ("get", FunctionTransformer(_get_title, validate=False)),
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=5000,
            sublinear_tf=True,
            lowercase=True,
        )),
    ])

    title_char = Pipeline([
        ("get", FunctionTransformer(_get_title, validate=False)),
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=5000,
            sublinear_tf=True,
            lowercase=True,
        )),
    ])

    desc_word = Pipeline([
        ("get", FunctionTransformer(_get_description, validate=False)),
        ("tfidf", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            max_features=20000,
            sublinear_tf=True,
            stop_words="english",
            lowercase=True,
        )),
    ])

    numeric = Pipeline([
        ("get", FunctionTransformer(_get_numeric, validate=False, accept_sparse=True)),
        ("scale", MaxAbsScaler()),
    ])

    return FeatureUnion([
        ("title_word", title_word),
        ("title_char", title_char),
        ("desc_word",  desc_word),
        ("numeric",    numeric),
    ])
