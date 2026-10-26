"""Evaluate LIME deletion/insertion curves on the corrected English test split."""

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib

from backend.app.explanations import explain_text
from backend.app.faithfulness import area_under_curve, perturbation_curve
from backend.app.data_pipeline import load_records


def _ranked_indices(text: str, segments: list[dict[str, object]]) -> tuple[list[str], list[int]]:
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    ranked: list[int] = []
    used: set[int] = set()
    for segment in segments:
        value = str(segment["text"])
        for index, token in enumerate(tokens):
            if index not in used and token == value:
                ranked.append(index)
                used.add(index)
                break
    return tokens, ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LIME perturbation faithfulness.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/nvd_records.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits/cve_splits.json"))
    parser.add_argument("--model", type=Path, default=Path("models/baseline/model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/faithfulness.json"))
    args = parser.parse_args()

    records = {record.cve_id: record for record in load_records(args.data)}
    test_ids = json.loads(args.splits.read_text(encoding="utf-8"))["splits"]["test"]
    model = joblib.load(args.model)
    rows: list[dict[str, object]] = []
    for cve_id in test_ids:
        text = records[cve_id].description
        explanation = explain_text(args.model, text, num_features=10, num_samples=500)
        tokens, ranked = _ranked_indices(text, explanation["segments"])
        predicted = explanation["prediction"]
        label_index = list(model.classes_).index(predicted)

        def probability(value: str) -> float:
            return float(model.predict_proba([value])[0][label_index])

        deletion = perturbation_curve(tokens, ranked, probability, "deletion")
        insertion = perturbation_curve(tokens, ranked, probability, "insertion")
        rows.append(
            {
                "cve_id": cve_id,
                "predicted_label": predicted,
                "matched_explanation_segments": len(ranked),
                "deletion_auc": area_under_curve(deletion),
                "insertion_auc": area_under_curve(insertion),
                "deletion_reference_probability": deletion[0]["probability"],
                "deletion_final_probability": deletion[-1]["probability"],
                "insertion_baseline_probability": insertion[0]["probability"],
                "insertion_final_probability": insertion[-1]["probability"],
            }
        )

    result = {
        "status": "evaluated_english_test_split",
        "language": "en",
        "sample_count": len(rows),
        "aggregation": {
            "mean_deletion_auc": sum(float(row["deletion_auc"]) for row in rows) / len(rows),
            "mean_insertion_auc": sum(float(row["insertion_auc"]) for row in rows) / len(rows),
            "mean_deletion_probability_drop": sum(
                float(row["deletion_reference_probability"]) - float(row["deletion_final_probability"])
                for row in rows
            ) / len(rows),
        },
        "settings": {
            "method": "LIME",
            "num_features": 10,
            "num_samples": 500,
            "tokenization": "unicode word-or-punctuation tokens; unmatched LIME segments are omitted",
            "deletion_baseline": "empty text",
            "insertion_baseline": "empty text",
        },
        "rows": rows,
        "provenance": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "split_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
            "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "test_cve_ids": test_ids,
            "dependency_versions": {"scikit-learn": version("scikit-learn"), "lime": version("lime")},
        },
        "limitations": [
            "Deletion and insertion curves are perturbation diagnostics, not causal proof.",
            "Only LIME segments that exactly match a token are perturbed.",
            "No translated-language faithfulness result is reported because no reliable translations are available.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["aggregation"], indent=2))


if __name__ == "__main__":
    main()
