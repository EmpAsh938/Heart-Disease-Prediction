from __future__ import annotations

import argparse
from pathlib import Path

from src.models.inference import predict_from_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions for a CSV input file")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model file")
    parser.add_argument("--input", type=str, required=True, help="Path to the CSV file with input features")
    parser.add_argument("--features", type=str, required=True, help="Path to the feature columns file")
    args = parser.parse_args()

    predictions = predict_from_csv(model_path=args.model, input_path=args.input, feature_columns=args.features)
    print(predictions)


if __name__ == "__main__":
    main()
