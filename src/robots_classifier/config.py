from dataclasses import dataclass
from typing import List, Tuple


DEFAULT_TARGET_BOTS: List[str] = [
    "googlebot",
    "bingbot",
    "yandex",
    "baiduspider",
    "gptbot",
    "chatgpt-user",
    "ccbot",
    "anthropic-ai",
    "amazonbot",
    "facebookexternalhit",
    "twitterbot",
    "pinterest",
    "instagram",
    "tiktok",
    "omgilibot",
]


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.2
    random_state: int = 42


@dataclass(frozen=True)
class TfidfConfig:
    ngram_range: Tuple[int, int] = (3, 4)
    max_features: int = 4000
    analyzer: str = "char"


@dataclass(frozen=True)
class LRConfig:
    max_iter: int = 1000
    solver: str = "saga"
    n_jobs: int = -1
    class_weight: str = "balanced"
    C: float = 1.0


@dataclass(frozen=True)
class RFConfig:
    n_estimators: int = 150
    random_state: int = 42
    n_jobs: int = -1