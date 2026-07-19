from __future__ import annotations

from typing import Any

import pandas as pd

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency
    shap = None


def explain_with_shap(model: Any, feature_frame: pd.DataFrame) -> Any:
    if shap is None:
        raise ImportError("shap is not installed")

    explainer = shap.Explainer(model, feature_frame)
    return explainer(feature_frame)
