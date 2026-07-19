from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DatasetSchema:
    target: str
    numerical_features: List[str]
    categorical_features: List[str]

    @property
    def feature_columns(self):
        return (
            self.numerical_features +
            self.categorical_features
        )

    @property
    def all_columns(self):
        return self.feature_columns + [self.target]
    



HEART_SCHEMA = DatasetSchema(

    target="target",

    numerical_features=[
        "age",
        "trestbps",
        "chol",
        "thalach",
        "oldpeak"
    ],

    categorical_features=[
        "sex",
        "cp",
        "fbs",
        "restecg",
        "exang",
        "slope",
        "ca",
        "thal"
    ]

)