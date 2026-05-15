import json
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_deep_model import build_deep_learning_pipeline, clip_feature_outliers, train_deep_learning_model


class DeepLearningTrainingTests(unittest.TestCase):
    def test_build_deep_learning_pipeline_uses_mlp_classifier(self):
        pipeline = build_deep_learning_pipeline(max_iter=25)

        self.assertIn("clf", pipeline.named_steps)
        self.assertEqual(pipeline.named_steps["clf"].__class__.__name__, "MLPClassifier")
        self.assertEqual(pipeline.named_steps["clf"].hidden_layer_sizes, (64, 32))
        self.assertIn("clip", pipeline.named_steps)

    def test_clip_feature_outliers_caps_values_to_app_slider_ranges(self):
        clipped = clip_feature_outliers(
            pd.DataFrame(
                [
                    {
                        "orbital_period": 2000,
                        "transit_duration": 80,
                        "transit_depth": 50000,
                        "planet_radius": 100,
                    }
                ]
            )
        )

        self.assertEqual(float(clipped["orbital_period"].iloc[0]), 1000)
        self.assertEqual(float(clipped["transit_duration"].iloc[0]), 30)
        self.assertEqual(float(clipped["transit_depth"].iloc[0]), 20000)
        self.assertEqual(float(clipped["planet_radius"].iloc[0]), 40)

    def test_train_deep_learning_model_writes_model_and_metrics(self):
        rows = []
        labels = ["CONFIRMED", "CANDIDATE", "FALSE POSITIVE"]
        for i, label in enumerate(labels):
            for j in range(8):
                rows.append(
                    {
                        "orbital_period": 2 + i * 20 + j,
                        "transit_duration": 1 + i + j * 0.1,
                        "transit_depth": 200 + i * 1500 + j * 10,
                        "planet_radius": 1 + i * 4 + j * 0.05,
                        "label": label,
                    }
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            data_path = tmp / "exoplanets.csv"
            model_path = tmp / "deep_model.joblib"
            metrics_path = tmp / "deep_metrics.json"
            pd.DataFrame(rows).to_csv(data_path, index=False)

            metrics = train_deep_learning_model(
                data_path=data_path,
                model_path=model_path,
                metrics_path=metrics_path,
                max_iter=50,
                early_stopping=False,
            )

            self.assertTrue(model_path.exists())
            self.assertTrue(metrics_path.exists())
            self.assertEqual(metrics["model_type"], "MLPClassifier")
            self.assertEqual(metrics["features"], ["orbital_period", "transit_duration", "transit_depth", "planet_radius"])
            loaded_model = joblib.load(model_path)
            self.assertEqual(loaded_model.named_steps["clf"].__class__.__name__, "MLPClassifier")
            saved_metrics = json.loads(metrics_path.read_text())
            self.assertEqual(saved_metrics["model_type"], "MLPClassifier")
