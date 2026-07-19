from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def train_test_split_dataframe(
    dataframe: pd.DataFrame,
    target_column: str = "target",
    test_size: float = 0.2,
    random_state: int = 42,
):
    frame = dataframe.copy()
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

    return X_train, X_test, y_train, y_test, feature_columns
