from __future__ import annotations

from typing import Dict, List

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def extract_numeric_features(text: str, target_bots: List[str]) -> Dict[str, float]:
    s = str(text).lower()
    feats: Dict[str, float] = {}

    feats["disallow_count"] = float(s.count("disallow:"))
    feats["allow_count"] = float(s.count("allow:"))
    feats["user_agent_count"] = float(s.count("user-agent:"))
    feats["sitemap_count"] = float(s.count("sitemap:"))

    total_rules = feats["disallow_count"] + feats["allow_count"]
    feats["disallow_ratio"] = feats["disallow_count"] / total_rules if total_rules > 0 else 0.0
    feats["allow_ratio"] = feats["allow_count"] / total_rules if total_rules > 0 else 0.0

    for bot in target_bots:
        feats[f"bot_{bot}"] = 1.0 if bot in s else 0.0

    feats["blocks_all"] = 1.0 if ("user-agent: *" in s and "disallow: /" in s) else 0.0
    return feats


def build_numeric_feature_matrix(texts: pd.Series, target_bots: List[str]):
    # Returns (X_numeric_scaled, scaler, columns)
    feats = texts.apply(lambda t: extract_numeric_features(t, target_bots))
    df_num = pd.DataFrame(feats.tolist()).fillna(0.0)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(df_num.values)
    return X, scaler, list(df_num.columns)


def align_numeric_features(df_num: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    # Ensure columns match training columns (predict-time safety)
    for col in columns:
        if col not in df_num.columns:
            df_num[col] = 0.0
    return df_num[columns].fillna(0.0)