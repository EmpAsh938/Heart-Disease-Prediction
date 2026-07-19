from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_target_distribution(dataframe: pd.DataFrame, target_column: str = "target") -> None:
    sns.countplot(data=dataframe, x=target_column)
    plt.title("Target Distribution")
    plt.show()


def plot_correlation_heatmap(dataframe: pd.DataFrame) -> None:
    numeric_frame = dataframe.select_dtypes(include=["number"])
    correlation = numeric_frame.corr()
    sns.heatmap(correlation, annot=False, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()
