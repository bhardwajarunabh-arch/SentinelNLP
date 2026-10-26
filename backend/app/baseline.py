from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .metrics import classification_metrics


def build_baseline(class_weight: str | dict[str, float] | None = "balanced") -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1, sublinear_tf=True)),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            ("classifier", LogisticRegression(max_iter=1000, class_weight=class_weight)),
        ]
    )


def train_baseline(
    texts: list[str], labels: list[str], output_dir: Path, seed: int = 42
) -> dict[str, Any]:
    model = build_baseline()
    model.named_steps["classifier"].random_state = seed
    model.fit(texts, labels)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    metadata = {"model": "tfidf-word-char-logistic-regression", "seed": seed, "classes": list(model.classes_)}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def predict(model_path: Path, texts: list[str]) -> tuple[list[str], Any]:
    model = joblib.load(model_path)
    return model.predict(texts).tolist(), model.predict_proba(texts)


def evaluate(model_path: Path, texts: list[str], labels: list[str]) -> dict[str, Any]:
    predictions, probabilities = predict(model_path, texts)
    return classification_metrics(labels, predictions, probabilities)
