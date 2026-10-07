from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    macro_f1: float
    report: str


def compute_metrics(y_true, y_pred) -> Metrics:
    acc = float(accuracy_score(y_true, y_pred))
    f1m = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    report = classification_report(y_true, y_pred, zero_division=0)
    return Metrics(accuracy=acc, macro_f1=f1m, report=report)


def topk_probabilities(classes, proba_row, k: int = 5):
    pairs = list(zip(classes, proba_row))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[:k]