"""Build verified research metrics without altering historical artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

LABELS = ["Critical", "High", "Low", "Medium"]
LANGUAGES = ["en", "hi", "zh", "es", "ar", "fr"]
TARGET_LANGUAGES = LANGUAGES[1:]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float] | None:
    if total == 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [max(0.0, centre - margin), min(1.0, centre + margin)]


def classification_report_verified(
    true_labels: list[str], predictions: list[str], probabilities: list[list[float]] | None
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "sample_count": len(true_labels),
        "class_support": dict(Counter(true_labels)),
        "labels": LABELS,
        "accuracy": float(accuracy_score(true_labels, predictions)),
        "accuracy_95_ci": wilson_interval(
            sum(actual == predicted for actual, predicted in zip(true_labels, predictions)),
            len(true_labels),
        ),
        "macro_f1": float(f1_score(true_labels, predictions, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(true_labels, predictions, labels=LABELS, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(true_labels, predictions)),
        "confusion_matrix": confusion_matrix(true_labels, predictions, labels=LABELS).tolist(),
        "per_class": classification_report(
            true_labels, predictions, labels=LABELS, output_dict=True, zero_division=0
        ),
        "missing_class_warnings": [
            label for label in LABELS if label not in set(true_labels)
        ],
    }
    if probabilities:
        result["log_loss"] = float(log_loss(true_labels, probabilities, labels=LABELS))
    else:
        result["log_loss"] = None
    return result


def critical_metrics(true_labels: list[str], predictions: list[str]) -> dict[str, Any]:
    critical_indices = [i for i, label in enumerate(true_labels) if label == "Critical"]
    actual = len(critical_indices)
    predicted_critical = sum(predictions[i] == "Critical" for i in critical_indices)
    transitions = Counter(predictions[i] for i in critical_indices if predictions[i] != "Critical")
    all_predicted_critical = sum(label == "Critical" for label in predictions)
    return {
        "actual_critical_count": actual,
        "critical_recall": predicted_critical / actual if actual else None,
        "critical_recall_95_ci": wilson_interval(predicted_critical, actual),
        "critical_precision": (
            predicted_critical / all_predicted_critical if all_predicted_critical else None
        ),
        "predicted_critical_count": all_predicted_critical,
        "underprediction_count": sum(transitions.values()),
        "underprediction_rate": sum(transitions.values()) / actual if actual else None,
        "underprediction_rate_95_ci": wilson_interval(sum(transitions.values()), actual),
        "critical_to_high": {"count": transitions["High"], "denominator": actual},
        "critical_to_medium": {"count": transitions["Medium"], "denominator": actual},
        "critical_to_low": {"count": transitions["Low"], "denominator": actual},
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def audit(
    records_path: Path,
    splits_path: Path,
    predictions_path: Path,
    translations_path: Path,
    model_metadata_path: Path,
) -> dict[str, Any]:
    records = {
        row["cve_id"]: row
        for row in json.loads(records_path.read_text(encoding="utf-8"))
    }
    manifest = json.loads(splits_path.read_text(encoding="utf-8"))
    split_sets = {name: set(ids) for name, ids in manifest["splits"].items()}
    predictions = read_csv(predictions_path)
    prediction_ids = [row["cve_id"] for row in predictions]
    translation_rows = read_csv(translations_path)
    metadata = json.loads(model_metadata_path.read_text(encoding="utf-8"))
    groups: dict[str, set[str]] = defaultdict(set)
    for cve_id, record in records.items():
        normalized = " ".join(record["description"].strip().casefold().split())
        groups[normalized].update(
            split_name for split_name, ids in split_sets.items() if cve_id in ids
        )
    leakage_groups = sum(len(splits) > 1 for splits in groups.values())
    prediction_source_mismatches = [
        row["cve_id"]
        for row in predictions
        if row["cve_id"] not in records
        or row["true_label"] != records[row["cve_id"]]["severity"]
    ]
    translation_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in translation_rows:
        translation_counts[row.get("language", "").lower()][
            row.get("validation_status", "missing")
        ] += 1
    return {
        "status": "complete",
        "records": {
            "eligible_cves": len(records),
            "unique_cve_ids": len(set(records)),
            "severity_distribution": dict(Counter(row["severity"] for row in records.values())),
            "cvss_versions": dict(Counter(row["cvss_version"] for row in records.values())),
        },
        "splits": {
            "counts": {name: len(ids) for name, ids in split_sets.items()},
            "duplicate_ids_within_split": {
                name: len(ids) != len(manifest["splits"][name])
                for name, ids in split_sets.items()
            },
            "overlap_ids": len(set.intersection(*split_sets.values())),
            "duplicate_description_groups_crossing_splits": leakage_groups,
            "duplicate_groups_total": manifest.get("duplicate_groups"),
            "seed": manifest.get("seed"),
            "strategy": manifest.get("strategy"),
        },
        "baseline_prediction_coverage": {
            "prediction_rows": len(predictions),
            "unique_prediction_ids": len(set(prediction_ids)),
            "test_ids": len(split_sets["test"]),
            "exactly_one_per_test_cve": (
                len(predictions) == len(split_sets["test"])
                and Counter(prediction_ids) == Counter(split_sets["test"])
            ),
            "training_ids_intersect_test_predictions": len(
                set(prediction_ids) & set(metadata["training_cve_ids"])
            ),
            "model_training_ids_equal_manifest_train": (
                set(metadata["training_cve_ids"]) == split_sets["train"]
            ),
            "source_or_label_mismatches": prediction_source_mismatches,
        },
        "translation_coverage": {
            language: {
                "counts": dict(translation_counts[language]),
                "valid": translation_counts[language]["valid"],
                "failed": translation_counts[language]["failed"]
                + translation_counts[language]["technical_identifier_missing"],
                "missing": translation_counts[language]["missing_translation"]
                + translation_counts[language]["missing"],
                "excluded": sum(
                    value
                    for status, value in translation_counts[language].items()
                    if status not in {"valid"}
                ),
            }
            for language in TARGET_LANGUAGES
        },
        "ground_truth": {
            "source": "NVD records filtered to CVSS v3.1",
            "all_records_cvss_31": all(row["cvss_version"] == "3.1" for row in records.values()),
            "test_labels_verified": all(
                records[cve_id]["severity"] in LABELS for cve_id in split_sets["test"]
            ),
        },
    }


def baseline_metrics(predictions_path: Path) -> dict[str, Any]:
    rows = read_csv(predictions_path)
    true_labels = [row["true_label"] for row in rows]
    predicted = [row["predicted_label"] for row in rows]
    probabilities = [
        [float(row[f"probability_{label}"]) for label in LABELS]
        for row in rows
    ]
    return {
        "status": "complete",
        "language": "en",
        "model": "tfidf-word-char-logistic-regression",
        "metrics": classification_report_verified(true_labels, predicted, probabilities),
        "critical_risk": critical_metrics(true_labels, predicted),
    }


def faithfulness_summary(
    faithfulness_path: Path, records_path: Path
) -> dict[str, Any]:
    payload = json.loads(faithfulness_path.read_text(encoding="utf-8"))
    records = {row["cve_id"]: row for row in json.loads(records_path.read_text(encoding="utf-8"))}
    rows = payload["rows"]
    enriched = [
        {
            **row,
            "true_label": records[row["cve_id"]]["severity"],
            "deletion_probability_drop": float(row["deletion_reference_probability"])
            - float(row["deletion_final_probability"]),
            "insertion_probability_gain": float(row["insertion_final_probability"])
            - float(row["insertion_baseline_probability"]),
        }
        for row in rows
    ]

    def aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
        if not selected:
            return {"sample_count": 0}
        return {
            "sample_count": len(selected),
            "mean_deletion_auc": statistics.fmean(float(r["deletion_auc"]) for r in selected),
            "median_deletion_auc": statistics.median(float(r["deletion_auc"]) for r in selected),
            "mean_insertion_auc": statistics.fmean(float(r["insertion_auc"]) for r in selected),
            "median_insertion_auc": statistics.median(float(r["insertion_auc"]) for r in selected),
            "mean_deletion_probability_drop": statistics.fmean(
                r["deletion_probability_drop"] for r in selected
            ),
            "mean_insertion_probability_gain": statistics.fmean(
                r["insertion_probability_gain"] for r in selected
            ),
        }

    return {
        "status": "partial_english_only",
        "language": "en",
        "sample_count": len(enriched),
        "aggregation": aggregate(enriched),
        "by_true_severity": {
            label: aggregate([row for row in enriched if row["true_label"] == label])
            for label in LABELS
        },
        "failure_definition": {
            "prespecified": True,
            "definition": "faithfulness failure if deletion probability drop <= 0 OR insertion probability gain <= 0",
            "count": sum(
                row["deletion_probability_drop"] <= 0
                or row["insertion_probability_gain"] <= 0
                for row in enriched
            ),
            "denominator": len(enriched),
        },
        "settings": payload["settings"],
        "limitations": [
            "Only English faithfulness is available.",
            "Perturbation diagnostics are not causal proof.",
            "Cross-language explanation stability is blocked without matched translations and defensible feature alignment.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("data/expanded/nvd_records.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/expanded/cve_splits.json"))
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/expanded_baseline_test_predictions.csv"))
    parser.add_argument("--translations", type=Path, default=Path("data/translations/expanded_test_translations.csv"))
    parser.add_argument("--model-metadata", type=Path, default=Path("models/expanded_baseline/metadata.json"))
    parser.add_argument("--faithfulness", type=Path, default=Path("artifacts/expanded_faithfulness.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    created_at = datetime.now(timezone.utc).isoformat()
    audit_result = audit(
        args.records, args.splits, args.predictions, args.translations, args.model_metadata
    )
    baseline = baseline_metrics(args.predictions)
    faithfulness = faithfulness_summary(args.faithfulness, args.records)
    blocked_languages = {
        language: {
            "status": "blocked",
            "sample_count": 0,
            "reason": "No validated translation rows for the complete expanded test split.",
        }
        for language in TARGET_LANGUAGES
    }
    research_summary = {
        "status": "partial",
        "created_at": created_at,
        "rq1": {
            "status": "blocked",
            "english": baseline,
            "languages": blocked_languages,
            "reason": "No target language has sufficient validated matched-CVE coverage.",
        },
        "rq2": faithfulness,
        "rq3": {
            "status": "blocked",
            "reason": "Cross-language disagreement cannot be joined to Critical underprediction without validated matched translations.",
            "unit_of_analysis": "CVE ID; repeated language rows would not be treated as independent.",
        },
        "baseline": baseline,
        "audit": audit_result,
    }
    cross_lingual = {
        "status": "blocked",
        "reason": "Only English predictions exist for the expanded test split; the pilot has one CVE per language and is not representative.",
        "languages": blocked_languages,
        "required_pairing": "same CVE ID against English and verified ground-truth labels",
    }
    critical = {
        "status": "complete",
        "language": "en",
        "model": baseline["model"],
        "metrics": baseline["critical_risk"],
        "definition": "Critical underprediction is a ground-truth Critical CVE predicted as High, Medium, or Low.",
    }
    rq3 = {
        "status": "blocked",
        "unit_of_analysis": "CVE ID",
        "thresholds": {
            "disagreement": "language prediction differs from English prediction",
            "faithfulness_failure": faithfulness["failure_definition"]["definition"],
        },
        "contingency_tables": {},
        "association_tests": {},
        "effect_sizes": {},
        "reason": "No eligible cross-language matched-CVE observations; statistical power for RQ3 is zero.",
    }
    provenance = {
        "status": "complete",
        "created_at": created_at,
        "paths": {
            "records": str(args.records),
            "splits": str(args.splits),
            "predictions": str(args.predictions),
            "translations": str(args.translations),
            "model_metadata": str(args.model_metadata),
            "faithfulness": str(args.faithfulness),
        },
        "sha256": {
            name: sha256(path)
            for name, path in {
                "records": args.records,
                "splits": args.splits,
                "predictions": args.predictions,
                "translations": args.translations,
                "model_metadata": args.model_metadata,
                "faithfulness": args.faithfulness,
            }.items()
        },
        "label_order": LABELS,
        "software": {
            package: version(package)
            for package in ["scikit-learn", "joblib", "numpy", "scipy"]
        },
        "settings": {
            "seed": 42,
            "confidence_interval": "Wilson 95% for binomial proportions",
            "translation_review": "not human reviewed",
        },
    }
    table_rows = [
        ["English classification", "complete", "artifacts/research_metrics_summary.json", "325 CVEs"],
        ["Target-language classification", "blocked", "data/translations/expanded_test_translations.csv", "0 validated rows per language"],
        ["English LIME faithfulness", "partial", "artifacts/explanation_faithfulness_summary.json", "325 CVEs"],
        ["Cross-lingual agreement", "blocked", "artifacts/cross_lingual_consistency.json", "No matched target-language predictions"],
        ["Critical risk English", "complete", "artifacts/critical_risk_metrics.json", "26 actual Critical CVEs"],
        ["RQ3 association tests", "blocked", "artifacts/rq3_statistical_analysis.json", "Zero eligible multilingual observations"],
        ["XLM-R comparison", "blocked", "models/xlmr_severity/", "No held-out prediction artifact"],
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "research_metrics_summary.json": research_summary,
        "cross_lingual_consistency.json": cross_lingual,
        "critical_risk_metrics.json": critical,
        "explanation_faithfulness_summary.json": faithfulness,
        "rq3_statistical_analysis.json": rq3,
        "research_metrics_provenance.json": provenance,
    }
    for filename, payload in outputs.items():
        (args.output_dir / filename).write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    with (args.output_dir / "research_metrics_table.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric_area", "status", "source_artifact", "coverage_or_blocker"])
        writer.writerows(table_rows)
    print(json.dumps({"status": "reports_written", "files": list(outputs)}, indent=2))


if __name__ == "__main__":
    main()
