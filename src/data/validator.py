from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.schema import HEART_SCHEMA


def validate_dataframe(dataframe: pd.DataFrame, schema=HEART_SCHEMA) -> pd.DataFrame:
    frame = dataframe.copy()

    missing_columns = [column for column in schema.all_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    for column in frame.columns:
        if frame[column].isna().any():
            if pd.api.types.is_numeric_dtype(frame[column]):
                frame[column] = frame[column].fillna(frame[column].median())
            else:
                frame[column] = frame[column].fillna(frame[column].mode().iloc[0])

    for column in schema.numerical_features:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    if schema.target in frame.columns:
        frame[schema.target] = pd.to_numeric(frame[schema.target], errors="coerce")
        if frame[schema.target].isna().any():
            raise ValueError("Target column contains non-numeric values")

    return frame


def validate_csv(path: str | Path, schema=HEART_SCHEMA) -> pd.DataFrame:
    return validate_dataframe(pd.read_csv(path), schema=schema)
