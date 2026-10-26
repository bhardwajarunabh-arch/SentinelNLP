from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib


def explain_text(model_path: Path, text: str, num_features: int = 10, num_samples: int = 500) -> dict[str, Any]:
    try:
        from lime.lime_text import LimeTextExplainer
    except ImportError as exc:
        raise RuntimeError("LIME is not installed; install backend requirements to enable explanations") from exc
    model = joblib.load(model_path)
    explainer = LimeTextExplainer(class_names=list(model.classes_))
    predicted_label = model.predict([text])[0]
    label_index = list(model.classes_).index(predicted_label)
    explanation = explainer.explain_instance(
        text,
        model.predict_proba,
        labels=[label_index],
        num_features=num_features,
        num_samples=num_samples,
    )
    return {
        "prediction": predicted_label,
        "segments": [
            {"text": segment, "weight": float(weight), "rank": rank}
            for rank, (segment, weight) in enumerate(explanation.as_list(label=label_index))
        ],
        "method": "LIME",
        "model": "tfidf-word-char-logistic-regression",
        "settings": {"num_features": num_features, "num_samples": num_samples},
        "limitations": "LIME is a local surrogate explanation and does not establish causal importance.",
    }
