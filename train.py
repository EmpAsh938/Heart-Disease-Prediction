from __future__ import annotations

import argparse
from pathlib import Path

from src.models.trainer import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the heart disease prediction model")
    parser.add_argument("--data", type=str, default="data/raw/heart_disease_data.csv", help="Path to the training CSV")
    parser.add_argument("--model", type=str, default="random_forest", choices=["logistic_regression", "random_forest"], help="Model to train")
    parser.add_argument("--output-dir", type=str, default="models", help="Directory for model artifacts")
    args = parser.parse_args()

    result = train_model(data_path=args.data, model_name=args.model, output_dir=args.output_dir)
    print(f"Model saved to: {result['model_path']}")
    print(f"Metrics saved to: {result['metrics_path']}")
    print(f"Feature columns saved to: {result['feature_columns_path']}")


if __name__ == "__main__":
    main()
