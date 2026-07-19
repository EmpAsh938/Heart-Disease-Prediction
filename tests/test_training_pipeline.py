import json
import tempfile
import unittest
from pathlib import Path

from src.models.inference import predict_from_frame
from src.models.trainer import train_model


class TrainingPipelineTest(unittest.TestCase):
    def test_train_and_predict_round_trip(self):
        repo_root = Path(__file__).resolve().parent.parent
        data_path = repo_root / "data" / "raw" / "heart_disease_data.csv"

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = train_model(data_path=data_path, model_name="random_forest", output_dir=output_dir)

            self.assertTrue(result["model_path"].exists())
            self.assertTrue(result["metrics_path"].exists())
            self.assertTrue(result["feature_columns_path"].exists())

            metrics = json.loads(result["metrics_path"].read_text())
            self.assertIn("accuracy", metrics)
            self.assertGreaterEqual(metrics["accuracy"], 0.0)

            sample_frame = result["feature_frame"].head(3).copy()
            predictions = predict_from_frame(
                model_path=result["model_path"],
                feature_frame=sample_frame,
                feature_columns=result["feature_columns_path"],
            )
            self.assertEqual(len(predictions), 3)


if __name__ == "__main__":
    unittest.main()
