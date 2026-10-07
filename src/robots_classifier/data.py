from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = {"site", "category", "robots_content"}


@dataclass(frozen=True)
class Dataset:
    df: pd.DataFrame
    name: str

    @property
    def n_rows(self) -> int:
        return len(self.df)

    @property
    def n_classes(self) -> int:
        return self.df["category"].nunique()


def load_dataset(csv_path: str | Path) -> Dataset:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing columns: {sorted(missing)}")

    df = df.copy()
    df["robots_content"] = df["robots_content"].fillna("").astype(str)
    df["category"] = df["category"].astype(str)

    return Dataset(df=df, name=csv_path.name)


def stratified_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["category"],
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)