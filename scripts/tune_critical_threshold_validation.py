"""Tune one shared Critical probability threshold on fixed validation examples only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.data_pipeline import load_records
from backend.app.metrics import classification_metrics
from backend.app.risk import critical_risk_metrics
from scripts.evaluate_xlmr_research import LANGUAGES, LABELS, sha256_file
from scripts.train_xlmr import prepare_multilingual_examples, sha256_ids

THRESHOLDS = [round(value, 2) for value in np.arange(0.20, 0.701, 0.05)]
SELECTION_RULE = (
    "Choose the threshold with highest mean per-language validation Macro-F1, "
    "subject to mean per-language Critical recall strictly above argmax and Low "
    "recall not below argmax in any language. Ties prefer higher mean Critical "
    "recall, then the higher threshold. If no threshold is eligible, retain "
    "standard argmax."
)


def critical_threshold_predictions(
    probabilities: np.ndarray, threshold: float, labels: list[str] = LABELS
) -> list[str]:
    if probabilities.ndim != 2 or probabilities.shape[1] != len(labels):
        raise ValueError("Probability matrix shape must match the label mapping")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Critical probability threshold must be between 0 and 1")
    critical_index = labels.index("Critical")
    non_critical_indices = [
        index for index in range(len(labels)) if index != critical_index
    ]
    predictions = []
    for row in probabilities:
        if not np.isfinite(row).all() or np.any(row < 0):
            raise ValueError("Probabilities must be finite and non-negative")
        if row[critical_index] >= threshold:
            predictions.append("Critical")
        else:
            best_index = max(non_critical_indices, key=lambda index: row[index])
            predictions.append(labels[best_index])
    return predictions


def select_shared_threshold(
    baseline: dict[str, Any], candidates: dict[str, dict[str, Any]]
) -> float | None:
    eligible = [
        float(threshold)
        for threshold, result in candidates.items()
        if result["mean_macro_f1"] > baseline["mean_macro_f1"]
        and result["mean_critical_recall"] > baseline["mean_critical_recall"]
        and all(
            result["per_language"][language]["low_recall"]
            >= baseline["per_language"][language]["low_recall"]
            for language in LANGUAGES
        )
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda threshold: (
            candidates[f"{threshold:.2f}"]["mean_macro_f1"],
            candidates[f"{threshold:.2f}"]["mean_critical_recall"],
            threshold,
        ),
    )


def metric_summary(
    examples: list[dict[str, Any]], predictions: list[str]
) -> dict[str, Any]:
    truth = [example["severity"] for example in examples]
    metrics = classification_metrics(truth, predictions, labels=LABELS)
    critical = metrics["per_class"]["Critical"]
    low = metrics["per_class"]["Low"]
    risk = critical_risk_metrics(truth, predictions)
    true_critical_count = risk["critical_ground_truth_count"]
    predicted_critical_count = sum(label == "Critical" for label in predictions)
    true_positive_critical = sum(
        true_label == predicted_label == "Critical"
        for true_label, predicted_label in zip(truth, predictions)
    )
    metrics["critical_class"] = {
        "precision": critical["precision"],
        "recall": critical["recall"],
        "f1": critical["f1-score"],
        "false_negative_count": risk["critical_false_negative_count"],
        "false_negative_rate": risk["critical_underprediction_rate"],
        "false_positive_count": predicted_critical_count - true_positive_critical,
    }
    metrics["low_recall"] = low["recall"]
    metrics.pop("ordinal", None)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("data/expanded/nvd_records.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/expanded/cve_splits.json"))
    parser.add_argument(
        "--translations",
        type=Path,
        default=Path(
            "artifacts/sentinelnlp_multilingual_xlmr_20261026_v1/"
            "translations/train_validation_translations.csv"
        ),
    )
    parser.add_argument(
        "--training-provenance",
        type=Path,
        default=Path(
            "artifacts/sentinelnlp_multilingual_xlmr_20261026_v1/"
            "training_v2/training_provenance.json"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "artifacts/sentinelnlp_multilingual_xlmr_20261026_v1/"
            "training_v2/checkpoint"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "artifacts/sentinelnlp_multilingual_xlmr_20261026_v1/"
            "validation_threshold_v1"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    started = time.perf_counter()
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for validation-only inference.")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir = output / "validation_predictions"
    prediction_dir.mkdir()

    records_by_id = {record.cve_id: record for record in load_records(args.records)}
    split_data = json.loads(args.splits.read_text(encoding="utf-8"))
    split = split_data["splits"]
    train_ids = split["train"]
    validation_ids = split["validation"]
    test_ids = split["test"]
    if (
        set(train_ids) & set(validation_ids)
        or set(train_ids) & set(test_ids)
        or set(validation_ids) & set(test_ids)
    ):
        raise ValueError("Fixed split contains overlapping CVE IDs")
    if set(validation_ids) & set(test_ids):
        raise ValueError("Test CVEs cannot enter validation threshold tuning")

    records_for_training = {
        cve_id: {
            "description": record.description,
            "severity": record.severity,
        }
        for cve_id, record in records_by_id.items()
    }
    _, validation_examples, translation_report = prepare_multilingual_examples(
        records_for_training, train_ids, validation_ids, args.translations
    )
    validation_id_set = set(validation_ids)
    if any(example["cve_id"] not in validation_id_set for example in validation_examples):
        raise ValueError("Prepared validation examples contain a non-validation CVE")
    if any(example["cve_id"] in set(test_ids) for example in validation_examples):
        raise ValueError("Prepared validation examples contain a test CVE")

    checkpoint = args.checkpoint.resolve()
    provenance = json.loads(args.training_provenance.read_text(encoding="utf-8"))
    if provenance["status"] != "trained" or provenance["test_ids_used_for_training_or_selection"]:
        raise ValueError("Training provenance is incomplete or reports test-set use")
    if provenance["split_id_sha256"]["validation"] != sha256_ids(validation_ids):
        raise ValueError("Validation IDs do not match the training provenance")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        str(checkpoint), local_files_only=True
    )
    expected_mapping = {str(index): label for index, label in enumerate(LABELS)}
    if {str(key): value for key, value in model.config.id2label.items()} != expected_mapping:
        raise ValueError("Checkpoint label mapping does not match the established mapping")
    model.to(torch.device("cuda"))
    model.eval()

    examples_by_language = {
        language: [
            example for example in validation_examples if example["language"] == language
        ]
        for language in LANGUAGES
    }
    probabilities_by_language: dict[str, np.ndarray] = {}
    for language, examples in examples_by_language.items():
        if not examples:
            raise ValueError(f"No valid fixed-validation examples for {language}")
        probability_batches = []
        for start in range(0, len(examples), args.batch_size):
            batch = examples[start : start + args.batch_size]
            encoded = tokenizer(
                [example["description"] for example in batch],
                return_tensors="pt",
                truncation=True,
                max_length=args.max_length,
                padding=True,
            )
            encoded = {key: value.to("cuda") for key, value in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded).logits
                probability_batches.append(
                    torch.softmax(logits.float(), dim=-1).cpu().numpy()
                )
        probabilities_by_language[language] = np.concatenate(probability_batches)

    labels_by_language: dict[str, list[str]] = {}
    for language, examples in examples_by_language.items():
        probabilities = probabilities_by_language[language]
        standard_predictions = [
            model.config.id2label[int(index)]
            for index in probabilities.argmax(axis=1)
        ]
        labels_by_language[language] = standard_predictions
        prediction_path = prediction_dir / f"{language}_validation_predictions.csv"
        with prediction_path.open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "cve_id",
                "language",
                "true_label",
                "predicted_label",
                *[f"probability_{label}" for label in LABELS],
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for example, probability, prediction in zip(
                examples, probabilities, standard_predictions
            ):
                writer.writerow(
                    {
                        "cve_id": example["cve_id"],
                        "language": language,
                        "true_label": example["severity"],
                        "predicted_label": prediction,
                        **{
                            f"probability_{label}": float(probability[index])
                            for index, label in enumerate(LABELS)
                        },
                    }
                )

    def summarize_operating_point(threshold: float | None) -> dict[str, Any]:
        per_language = {}
        for language, examples in examples_by_language.items():
            predictions = (
                labels_by_language[language]
                if threshold is None
                else critical_threshold_predictions(
                    probabilities_by_language[language], threshold
                )
            )
            per_language[language] = metric_summary(examples, predictions)
        return {
            "threshold": "argmax" if threshold is None else threshold,
            "per_language": per_language,
            "mean_macro_f1": float(
                np.mean([metric["macro_f1"] for metric in per_language.values()])
            ),
            "mean_critical_recall": float(
                np.mean(
                    [metric["critical_class"]["recall"] for metric in per_language.values()]
                )
            ),
        }

    baseline = summarize_operating_point(None)
    candidates = {
        f"{threshold:.2f}": summarize_operating_point(threshold)
        for threshold in THRESHOLDS
    }
    selected_threshold = select_shared_threshold(baseline, candidates)
    selected = (
        baseline
        if selected_threshold is None
        else candidates[f"{selected_threshold:.2f}"]
    )
    selected_prediction_dir = output / "selected_predictions"
    selected_prediction_dir.mkdir()
    for language, examples in examples_by_language.items():
        probabilities = probabilities_by_language[language]
        threshold_predictions = (
            labels_by_language[language]
            if selected_threshold is None
            else critical_threshold_predictions(probabilities, selected_threshold)
        )
        prediction_path = (
            selected_prediction_dir / f"{language}_validation_predictions.csv"
        )
        with prediction_path.open("w", newline="", encoding="utf-8") as handle:
            fields = [
                "cve_id",
                "language",
                "true_label",
                "argmax_predicted_label",
                "threshold_predicted_label",
                *[f"probability_{label}" for label in LABELS],
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for example, probability, argmax_prediction, threshold_prediction in zip(
                examples,
                probabilities,
                labels_by_language[language],
                threshold_predictions,
            ):
                writer.writerow(
                    {
                        "cve_id": example["cve_id"],
                        "language": language,
                        "true_label": example["severity"],
                        "argmax_predicted_label": argmax_prediction,
                        "threshold_predicted_label": threshold_prediction,
                        **{
                            f"probability_{label}": float(probability[index])
                            for index, label in enumerate(LABELS)
                        },
                    }
                )
    (output / "threshold_sweep.json").write_text(
        json.dumps(
            {
                "selection_rule": SELECTION_RULE,
                "threshold_grid": THRESHOLDS,
                "standard_argmax": baseline,
                "candidates": candidates,
                "selected_threshold": selected_threshold,
                "selected_operating_point": selected,
                "interpretation": (
                    "No swept threshold improved mean Macro-F1 while meeting the "
                    "Critical-recall and per-language Low-recall constraints."
                    if selected_threshold is None
                    else "Selected using fixed validation data only."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "provenance.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
                "records_sha256": sha256_file(args.records),
                "splits_sha256": sha256_file(args.splits),
                "validation_id_sha256": sha256_ids(validation_ids),
                "translation_sha256": sha256_file(args.translations),
                "translation_report": {
                    "eligible_validation_examples": len(validation_examples),
                    "counts_by_language": dict(
                        Counter(example["language"] for example in validation_examples)
                    ),
                    "excluded_row_count": len(translation_report["excluded_rows"]),
                    "test_cves_used": translation_report["test_cves_used"],
                },
                "training_provenance_sha256": sha256_file(args.training_provenance),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": {
                    path.name: sha256_file(path)
                    for path in checkpoint.iterdir()
                    if path.is_file()
                },
                "label_order": LABELS,
                "language_order": LANGUAGES,
                "threshold_grid": THRESHOLDS,
                "threshold_rule": (
                    "Predict Critical if P(Critical) >= threshold; otherwise choose "
                    "the highest-probability non-Critical class."
                ),
                "selection_rule": SELECTION_RULE,
                "seed": args.seed,
                "batch_size": args.batch_size,
                "max_length": args.max_length,
                "device": torch.cuda.get_device_name(0),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "runtime_seconds": time.perf_counter() - started,
                "test_data_used": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "run_summary.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "validation_samples_by_language": {
                    language: len(examples)
                    for language, examples in examples_by_language.items()
                },
                "standard_argmax_mean_macro_f1": baseline["mean_macro_f1"],
                "standard_argmax_mean_critical_recall": baseline[
                    "mean_critical_recall"
                ],
                "selected_threshold": selected_threshold,
                "selected_mean_macro_f1": selected["mean_macro_f1"],
                "selected_mean_critical_recall": selected["mean_critical_recall"],
                "selected_low_recall_by_language": {
                    language: metric["low_recall"]
                    for language, metric in selected["per_language"].items()
                },
                "test_data_used": False,
                "runtime_seconds": time.perf_counter() - started,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
