from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from joblib import dump, load
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from .config import LRConfig, RFConfig, TfidfConfig
from .features import build_numeric_feature_matrix, align_numeric_features, extract_numeric_features


@dataclass
class LRBundle:
    vectorizer: TfidfVectorizer
    model: LogisticRegression


@dataclass
class RFBundle:
    vectorizer: TfidfVectorizer
    model: RandomForestClassifier
    scaler: MinMaxScaler
    numeric_columns: List[str]
    target_bots: List[str]


def train_lr(
    train_texts: pd.Series,
    train_labels: pd.Series,
    tfidf_cfg: TfidfConfig,
    lr_cfg: LRConfig,
) -> LRBundle:
    vec = TfidfVectorizer(
        analyzer=tfidf_cfg.analyzer,
        ngram_range=tfidf_cfg.ngram_range,
        max_features=tfidf_cfg.max_features,
    )
    Xtr = vec.fit_transform(train_texts)

    clf = LogisticRegression(
        max_iter=lr_cfg.max_iter,
        solver=lr_cfg.solver,
        #n_jobs=lr_cfg.n_jobs,
        class_weight=lr_cfg.class_weight,
        C=lr_cfg.C,
    )
    clf.fit(Xtr, train_labels)
    return LRBundle(vectorizer=vec, model=clf)


def train_rf(
    train_texts: pd.Series,
    train_labels: pd.Series,
    tfidf_cfg: TfidfConfig,
    rf_cfg: RFConfig,
    max_features_text: int,
    target_bots: List[str],
) -> RFBundle:
    # Text part
    vec = TfidfVectorizer(
        analyzer=tfidf_cfg.analyzer,
        ngram_range=tfidf_cfg.ngram_range,
        max_features=max_features_text,
    )
    X_text = vec.fit_transform(train_texts)

    # Numeric part
    X_num, scaler, columns = build_numeric_feature_matrix(train_texts, target_bots)

    X = hstack([X_text, csr_matrix(X_num)])

    rf = RandomForestClassifier(
        n_estimators=rf_cfg.n_estimators,
        random_state=rf_cfg.random_state,
        n_jobs=rf_cfg.n_jobs,
    )
    rf.fit(X, train_labels)

    return RFBundle(
        vectorizer=vec,
        model=rf,
        scaler=scaler,
        numeric_columns=columns,
        target_bots=target_bots,
    )


def predict_lr(bundle: LRBundle, texts: List[str]):
    X = bundle.vectorizer.transform([str(t) for t in texts])
    pred = bundle.model.predict(X)
    proba = bundle.model.predict_proba(X) if hasattr(bundle.model, "predict_proba") else None
    return pred, proba


def predict_rf(bundle: RFBundle, texts: List[str]):
    X_text = bundle.vectorizer.transform([str(t) for t in texts])

    # Build numeric df aligned to training columns
    feats = [extract_numeric_features(t, bundle.target_bots) for t in texts]
    df_num = pd.DataFrame(feats)
    df_num = align_numeric_features(df_num, bundle.numeric_columns)
    X_num = bundle.scaler.transform(df_num.values)

    X = hstack([X_text, csr_matrix(X_num)])
    pred = bundle.model.predict(X)
    proba = bundle.model.predict_proba(X) if hasattr(bundle.model, "predict_proba") else None
    return pred, proba


def save_bundle(bundle, path: str):
    dump(bundle, path)


def load_bundle(path: str):
    return load(path)