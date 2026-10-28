import argparse
import csv
import hashlib
import json
import sys
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Permit ``python scripts\evaluate_translations.py`` from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.baseline import predict
from backend.app.data_pipeline import load_records
from backend.app.metrics import agreement_metrics, classification_metrics
from backend.app.risk import critical_risk_metrics
from backend.app.translations import validate_translation_for_language
LABELS = ["Critical", "High", "Low", "Medium"]


def evaluate_translations(
    rows: list[dict[str, str]],
    model_path: Path,
    records_path: Path,
    splits_path: Path,
    predictions_output: Path | None = None,
    translation_path: Path | None = None,
) -> dict[str, Any]:
    records = {record.cve_id: record for record in load_records(records_path)}
    split_manifest = json.loads(splits_path.read_text(encoding="utf-8"))
    test_ids = set(split_manifest["splits"]["test"])
    valid_rows: list[dict[str, str]] = []
    excluded_rows: dict[str, int] = {}
    excluded_pairs: list[dict[str, str]] = []
    validation_status_disagreements: list[dict[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    def exclude(pair: tuple[str, str], reason: str) -> None:
        excluded_rows[reason] = excluded_rows.get(reason, 0) + 1
        excluded_pairs.append({"cve_id": pair[0], "language": pair[1], "reason": reason})

    for row in rows:
        pair = (row.get("cve_id", ""), row.get("language", "").lower())
        if pair in seen_pairs:
            exclude(pair, "duplicate_cve_language")
            continue
        seen_pairs.add(pair)
        if row["cve_id"] not in records:
            raise ValueError(f"Translation CVE is missing from NVD records: {row['cve_id']}")
        if row["cve_id"] not in test_ids:
            exclude(pair, "not_in_test_split")
            continue
        if row["original_text"] != records[row["cve_id"]].description:
            exclude(pair, "source_mismatch")
            continue
        if row.get("severity") and row["severity"] != records[row["cve_id"]].severity:
            exclude(pair, "severity_mismatch")
            continue
        computed, _ = validate_translation_for_language(
            records[row["cve_id"]].description, row.get("translated_text", ""), row["language"].lower()
        )
        if computed != "valid":
            exclude(pair, computed)
            continue
        if row.get("validation_status") not in {"", "valid"}:
            validation_status_disagreements.append(
                {
                    "cve_id": pair[0],
                    "language": pair[1],
                    "stored_status": row["validation_status"],
                    "recomputed_status": computed,
                }
            )
        valid_rows.append(row)
    translation_review = {
        "status": "not_human_reviewed",
        "human_review_performed": False,
    }
    if not valid_rows:
        result = {
            "status": "blocked_no_reliable_translations",
            "sample_count": 0,
            "excluded_rows": excluded_rows,
            "excluded_pairs": excluded_pairs,
            "validation_status_disagreements": validation_status_disagreements,
            "translation_review": translation_review,
            "labels": LABELS,
            "languages": {
                language: {"sample_count": 0, "status": "missing"}
                for language in sorted({row["language"] for row in rows})
            },
            "provenance": {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "dataset_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
                "split_sha256": hashlib.sha256(splits_path.read_bytes()).hexdigest(),
                "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                "translation_sha256": (
                    hashlib.sha256(translation_path.read_bytes()).hexdigest()
                    if translation_path is not None
                    else None
                ),
                "test_cve_ids": sorted(test_ids),
                "dependency_versions": {"scikit-learn": version("scikit-learn"), "joblib": version("joblib")},
            },
            "limitation": "No reliable non-English translations are available for the current grouped test split.",
        }
        return result

    english_texts = [row["original_text"] for row in valid_rows]
    english_predictions, english_probabilities = predict(model_path, english_texts)
    english_labels = [records[row["cve_id"]].severity for row in valid_rows]
    result: dict[str, Any] = {
        "status": (
            "valid_train_only_test_translation_evaluation"
            if "not_in_test_split" not in excluded_rows
            else "partial_translation_evaluation_test_membership_filtered"
        ),
        "sample_count": len(valid_rows),
        "excluded_rows": excluded_rows,
        "excluded_pairs": excluded_pairs,
        "validation_status_disagreements": validation_status_disagreements,
        "translation_review": translation_review,
        "labels": LABELS,
        "english_severity": classification_metrics(
            english_labels, english_predictions, english_probabilities, labels=LABELS
        ),
        "english_critical_risk": critical_risk_metrics(english_labels, english_predictions),
        "agreement": agreement_metrics(english_predictions, english_predictions),
        "languages": {},
        "provenance": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_sha256": hashlib.sha256(records_path.read_bytes()).hexdigest(),
            "split_sha256": hashlib.sha256(splits_path.read_bytes()).hexdigest(),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "translation_sha256": None,
            "test_cve_ids": sorted({row["cve_id"] for row in valid_rows}),
            "dependency_versions": {"scikit-learn": version("scikit-learn"), "joblib": version("joblib")},
        },
    }
    all_english_predictions: list[str] = []
    prediction_rows: list[dict[str, str]] = []
    translated_predictions_by_language: dict[str, dict[str, str]] = {}
    for language in sorted({row["language"] for row in valid_rows}):
        language_rows = [row for row in valid_rows if row["language"] == language]
        indices = [
            index for index, row in enumerate(valid_rows) if row["language"] == language
        ]
        translated_predictions, translated_probabilities = predict(
            model_path, [row["translated_text"] for row in language_rows]
        )
        translated_predictions_by_language[language] = {
            row["cve_id"]: prediction
            for row, prediction in zip(language_rows, translated_predictions)
        }
        language_labels = [records[row["cve_id"]].severity for row in language_rows]
        result["languages"][language] = {
            "sample_count": len(language_rows),
            "excluded_count": sum(
                pair["language"] == language for pair in excluded_pairs
            ),
            "excluded_cve_ids": sorted(
                {pair["cve_id"] for pair in excluded_pairs if pair["language"] == language}
            ),
            "agreement": agreement_metrics(
                [english_predictions[index] for index in indices], translated_predictions
            ),
            "severity": classification_metrics(
                language_labels,
                translated_predictions,
                translated_probabilities,
                labels=LABELS,
            ),
            "critical_risk": critical_risk_metrics(language_labels, translated_predictions),
        }
        all_english_predictions.extend(english_predictions[index] for index in indices)
        prediction_rows.extend(
            {
                "cve_id": row["cve_id"],
                "language": language,
                "true_label": records[row["cve_id"]].severity,
                "english_predicted_label": english_predictions[index],
                "translated_predicted_label": translated_predictions[offset],
                "agreement": str(english_predictions[index] == translated_predictions[offset]),
            }
            for offset, (index, row) in enumerate(zip(indices, language_rows))
        )
    all_translated_predictions: list[str] = []
    for language in sorted(result["languages"]):
        language_rows = [row for row in valid_rows if row["language"] == language]
        predictions, _ = predict(model_path, [row["translated_text"] for row in language_rows])
        all_translated_predictions.extend(predictions)
    result["agreement"] = agreement_metrics(all_english_predictions, all_translated_predictions)
    paired_languages = sorted(translated_predictions_by_language)
    common_ids = (
        set.intersection(
            *(set(translated_predictions_by_language[language]) for language in paired_languages)
        )
        if paired_languages
        else set()
    )
    result["paired_cross_language"] = {
        "languages": paired_languages,
        "sample_count": len(common_ids),
        "cve_ids": sorted(common_ids),
        "excluded_cve_ids": sorted(
            {
                row["cve_id"]
                for row in excluded_pairs
                if row["language"] in paired_languages
            }
        ),
        "pairwise_prediction_agreement": {
            f"{left}:{right}": agreement_metrics(
                [translated_predictions_by_language[left][cve_id] for cve_id in sorted(common_ids)],
                [translated_predictions_by_language[right][cve_id] for cve_id in sorted(common_ids)],
            )
            for index, left in enumerate(paired_languages)
            for right in paired_languages[index + 1 :]
        },
    }
    if predictions_output is not None:
        predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with predictions_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "cve_id", "language", "true_label",
                    "english_predicted_label", "translated_predicted_label", "agreement",
                ],
            )
            writer.writeheader()
            writer.writerows(prediction_rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate paired translation predictions.")
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("models/baseline/model.joblib"))
    parser.add_argument("--records", type=Path, default=Path("data/processed/nvd_records.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits/cve_splits.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/metrics.json"))
    parser.add_argument("--predictions-output", type=Path)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.translations.open(encoding="utf-8", newline="")))
    result = evaluate_translations(
        rows,
        args.model,
        args.records,
        args.splits,
        args.predictions_output,
        args.translations,
    )
    result.setdefault("provenance", {})["translation_sha256"] = hashlib.sha256(
        args.translations.read_bytes()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
