from __future__ import annotations

import pandas as pd


def get_feature_importance(model, feature_columns: list[str]) -> pd.DataFrame:
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("The selected model does not expose feature_importances_")

    importance_values = model.feature_importances_
    return pd.DataFrame({"feature": feature_columns, "importance": importance_values}).sort_values("importance", ascending=False)
