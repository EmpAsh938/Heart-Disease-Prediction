from __future__ import annotations

import pandas as pd


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    frame = dataframe.copy()

    if "age" in frame.columns:
        frame["age_group"] = pd.cut(
            frame["age"],
            bins=[0, 40, 60, 100],
            labels=["young", "middle", "senior"],
            include_lowest=True,
        )

    if "oldpeak" in frame.columns:
        frame["oldpeak_flag"] = (frame["oldpeak"] > 2).astype(int)

    return frame
