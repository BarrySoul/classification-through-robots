from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class AnyModelBundle:
    """Serializable bundle: vectorizer + fitted model + metadata."""
    vectorizer: TfidfVectorizer
    model: Any
    model_name: str
    params: Dict[str, Any]