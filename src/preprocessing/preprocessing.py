from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def split_features_and_target(
    dataframe: pd.DataFrame,
    target_column: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, list[str], pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split the dataset into train/test features and labels."""
    frame = dataframe.copy()

    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    feature_columns = [column for column in frame.columns if column != target_column]
    X = frame[feature_columns]
    y = frame[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return X, y, feature_columns, X_train, X_test, y_train, y_test


def prepare_feature_frame(dataframe: pd.DataFrame, target_column: str = "target") -> Tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return feature dataframe, target series, and ordered feature names."""
    frame = dataframe.copy()
    if target_column not in frame.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe")

    feature_columns = [column for column in frame.columns if column != target_column]
    return frame[feature_columns], frame[target_column], feature_columns


def save_feature_columns(feature_columns: list[str], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(feature_columns), encoding="utf-8")
    return output
