from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


DATA = Path("data/exoplanets.csv")
OUT_MODEL = Path("artifacts/deep_model.joblib")
OUT_METRICS = Path("artifacts/deep_metrics.json")
FEATS = ["orbital_period", "transit_duration", "transit_depth", "planet_radius"]


def clip_feature_outliers(X):
    X = pd.DataFrame(X, columns=FEATS).copy()
    X["orbital_period"] = X["orbital_period"].clip(lower=0, upper=1000)
    X["transit_duration"] = X["transit_duration"].clip(lower=0, upper=30)
    X["transit_depth"] = X["transit_depth"].clip(lower=0, upper=20000)
    X["planet_radius"] = X["planet_radius"].clip(lower=0, upper=40)
    return X


def build_deep_learning_pipeline(max_iter: int = 220, early_stopping: bool = False) -> Pipeline:
    clipper = ColumnTransformer(
        [
            ("period", FunctionTransformer(np.clip, kw_args={"a_min": 0, "a_max": 1000}, validate=False), [0]),
            ("duration", FunctionTransformer(np.clip, kw_args={"a_min": 0, "a_max": 30}, validate=False), [1]),
            ("depth", FunctionTransformer(np.clip, kw_args={"a_min": 0, "a_max": 20000}, validate=False), [2]),
            ("radius", FunctionTransformer(np.clip, kw_args={"a_min": 0, "a_max": 40}, validate=False), [3]),
        ]
    )
    return Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("clip", clipper),
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    batch_size=256,
                    learning_rate_init=3e-4,
                    max_iter=max_iter,
                    early_stopping=early_stopping,
                    n_iter_no_change=12,
                    random_state=42,
                ),
            ),
        ]
    )


def train_deep_learning_model(
    data_path: str | Path = DATA,
    model_path: str | Path = OUT_MODEL,
    metrics_path: str | Path = OUT_METRICS,
    max_iter: int = 220,
    early_stopping: bool = False,
) -> dict:
    data_path = Path(data_path)
    model_path = Path(model_path)
    metrics_path = Path(metrics_path)

    df = pd.read_csv(data_path)
    X, y = df[FEATS], df["label"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    pipe = build_deep_learning_pipeline(max_iter=max_iter, early_stopping=early_stopping)
    pipe.fit(Xtr, ytr)
    yp = pipe.predict(Xte)

    rep = classification_report(yte, yp, output_dict=True, zero_division=0)
    cm = confusion_matrix(yte, yp).tolist()
    metrics = {
        "model_type": "MLPClassifier",
        "architecture": {
            "input_features": len(FEATS),
            "hidden_layers": [64, 32],
            "activation": "relu",
            "output_classes": sorted(df["label"].unique()),
        },
        "report": rep,
        "confusion_matrix": cm,
        "features": FEATS,
        "labels": sorted(df["label"].unique()),
    }

    os.makedirs(model_path.parent, exist_ok=True)
    joblib.dump(pipe, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    result = train_deep_learning_model()
    macro_f1 = result["report"]["macro avg"]["f1-score"]
    print(f"Deep Learning model: {OUT_MODEL}")
    print(f"Deep Learning metrics: {OUT_METRICS}")
    print("Architecture: MLPClassifier hidden layers [64, 32], ReLU activation")
    print(f"Macro F1: {macro_f1:.3f}")
