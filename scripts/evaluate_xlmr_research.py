"""Evaluate a fixed-split XLM-R checkpoint on English and validated test translations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.app.data_pipeline import load_records
from backend.app.metrics import agreement_metrics, classification_metrics
from backend.app.risk import critical_risk_metrics
from backend.app.translations import validate_translation_for_language

LABELS = ["Critical", "High", "Low", "Medium"]
LANGUAGES = ["en", "ar", "es", "fr", "hi", "zh"]
TARGET_LANGUAGES = ["ar", "es", "fr", "hi", "zh"]
ORDINAL_ORDER = ["Low", "Medium", "High", "Critical"]
BOOTSTRAP_REPLICATES = 1000


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def bootstrap_metric_intervals(
    truth: list[str], predictions: list[str], seed: int
) -> dict[str, list[float] | None]:
    if not truth:
        return {"accuracy_95_ci": None, "macro_f1_95_ci": None, "critical_fn_rate_95_ci": None}
    rng = np.random.default_rng(seed)
    truth_array = np.asarray(truth)
    prediction_array = np.asarray(predictions)
    scores: dict[str, list[float]] = {
        "accuracy": [],
        "macro_f1": [],
        "critical_fn_rate": [],
    }
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, len(truth_array), size=len(truth_array))
        sample_truth = truth_array[sample]
        sample_predictions = prediction_array[sample]
        scores["accuracy"].append(float(accuracy_score(sample_truth, sample_predictions)))
        scores["macro_f1"].append(
            float(
                f1_score(
                    sample_truth,
                    sample_predictions,
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                )
            )
        )
        critical_indices = sample_truth == "Critical"
        critical_count = int(critical_indices.sum())
        scores["critical_fn_rate"].append(
            float(np.sum(sample_predictions[critical_indices] != "Critical") / critical_count)
            if critical_count
            else 0.0
        )
    return {
        f"{name}_95_ci": [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]
        for name, values in scores.items()
    }


def paired_bootstrap_macro_f1_difference(
    truth: list[str], left: list[str], right: list[str], seed: int
) -> list[float] | None:
    if not truth:
        return None
    rng = np.random.default_rng(seed)
    truth_array = np.asarray(truth)
    left_array = np.asarray(left)
    right_array = np.asarray(right)
    differences = []
    for _ in range(BOOTSTRAP_REPLICATES):
        sample = rng.integers(0, len(truth_array), size=len(truth_array))
        differences.append(
            float(
                f1_score(truth_array[sample], left_array[sample], labels=LABELS, average="macro", zero_division=0)
                - f1_score(truth_array[sample], right_array[sample], labels=LABELS, average="macro", zero_division=0)
            )
        )
    return [float(np.quantile(differences, 0.025)), float(np.quantile(differences, 0.975))]


def paired_model_difference_metrics(
    truth: list[str],
    current_predictions: list[str],
    comparator_predictions: list[str],
    current_metrics: dict[str, Any],
    comparator_metrics: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "sample_count": len(truth),
        "macro_f1_difference_current_minus_comparator": (
            current_metrics["macro_f1"] - comparator_metrics["macro_f1"]
        ),
        "paired_macro_f1_difference_95_ci": paired_bootstrap_macro_f1_difference(
            truth, current_predictions, comparator_predictions, seed
        ),
        "critical_false_negative_difference_current_minus_comparator": (
            current_metrics["critical_class"]["false_negative_count"]
            - comparator_metrics["critical_class"]["false_negative_count"]
        ),
        "same_test_cve_ids": True,
    }


def predict_texts(
    model: Any, tokenizer: Any, texts: list[str], device: torch.device, batch_size: int, max_length: int
) -> tuple[list[str], list[list[float]]]:
    predicted_labels: list[str] = []
    probability_rows: list[list[float]] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        indices = probabilities.argmax(axis=1)
        predicted_labels.extend(model.config.id2label[int(index)] for index in indices)
        probability_rows.extend(probabilities.tolist())
    return predicted_labels, probability_rows


def metrics_for_rows(rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    truth = [row["true_label"] for row in rows]
    predicted = [row["predicted_label"] for row in rows]
    metrics = classification_metrics(truth, predicted, labels=LABELS)
    metrics["critical_risk"] = critical_risk_metrics(truth, predicted)
    metrics["critical_class"] = {
        "precision": metrics["per_class"]["Critical"]["precision"],
        "recall": metrics["per_class"]["Critical"]["recall"],
        "f1": metrics["per_class"]["Critical"]["f1-score"],
        "false_negative_count": metrics["critical_risk"]["critical_false_negative_count"],
        "false_negative_rate": metrics["critical_risk"]["critical_underprediction_rate"],
    }
    metrics["bootstrap_95_ci"] = bootstrap_metric_intervals(truth, predicted, seed)
    metrics["bootstrap_method"] = (
        f"CVE-level percentile bootstrap, {BOOTSTRAP_REPLICATES} replicates, seed {seed}"
    )
    return metrics


def read_prediction_csv(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    keyed: dict[str, dict[str, str]] = {}
    for row in rows:
        if row["cve_id"] in keyed:
            raise ValueError(f"Duplicate prediction CVE in {path}: {row['cve_id']}")
        keyed[row["cve_id"]] = row
    return keyed


def write_prediction_csv(
    output_dir: Path, language: str, rows: list[dict[str, Any]]
) -> Path:
    path = output_dir / "predictions" / f"{language}_test_predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def comparison_rows(
    model_metrics: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model_name, metric_map in model_metrics.items():
        for language in LANGUAGES:
            metrics = metric_map[language]
            ci = metrics["bootstrap_95_ci"]["macro_f1_95_ci"]
            rows.append(
                {
                    "language": language,
                    "model": model_name,
                    "sample_count": metrics["sample_count"],
                    "macro_f1": metrics["macro_f1"],
                    "macro_f1_95_ci_low": ci[0] if ci else "",
                    "macro_f1_95_ci_high": ci[1] if ci else "",
                    "critical_false_negative_count": metrics["critical_class"]["false_negative_count"],
                    "accuracy": metrics["accuracy"],
                    "weighted_f1": metrics["weighted_f1"],
                    "mcc": metrics["matthews_correlation_coefficient"],
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("data/expanded/nvd_records.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/expanded/cve_splits.json"))
    parser.add_argument(
        "--translations",
        type=Path,
        default=Path("data/translations/expanded_test_translations_generated.csv"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-english",
        type=Path,
        default=Path("artifacts/expanded_baseline_test_predictions.csv"),
    )
    parser.add_argument(
        "--baseline-translations",
        type=Path,
        default=Path("artifacts/sentinelnlp_translation_predictions_20261026.csv"),
    )
    parser.add_argument(
        "--comparison-prediction-dir",
        type=Path,
        action="append",
        default=[],
        help="Existing model prediction directory to compare on identical valid test CVEs.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exploratory-test-comparison",
        action="store_true",
        help="Mark test metrics exploratory when test results were previously observed.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for XLM-R test evaluation.")
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty experiment directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    prediction_dir = output / "predictions"
    prediction_dir.mkdir()
    experiment_name = output.parent.name if output.name.lower().startswith("evaluation") else output.name

    started = time.perf_counter()
    records = {record.cve_id: record for record in load_records(args.records)}
    split_manifest = json.loads(args.splits.read_text(encoding="utf-8"))
    test_ids = split_manifest["splits"]["test"]
    if len(test_ids) != len(set(test_ids)) or set(test_ids) - records.keys():
        raise ValueError("Fixed test split has duplicate or unknown CVE IDs")
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Fine-tuned checkpoint directory not found: {checkpoint}")
    device = torch.device("cuda")
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(checkpoint), local_files_only=True)
    if {str(key): value for key, value in model.config.id2label.items()} != {
        str(index): label for index, label in enumerate(LABELS)
    }:
        raise ValueError("Checkpoint label mapping does not match the established severity mapping")
    model.to(device)

    with args.translations.open(newline="", encoding="utf-8") as handle:
        translation_rows = list(csv.DictReader(handle))
    translation_by_pair: dict[tuple[str, str], dict[str, str]] = {}
    excluded_pairs: list[dict[str, str]] = []
    validation_status_disagreements: list[dict[str, str]] = []
    for row in translation_rows:
        cve_id = row.get("cve_id", "")
        language = row.get("language", "").lower()
        pair = (cve_id, language)
        if pair in translation_by_pair or cve_id not in records or cve_id not in test_ids:
            excluded_pairs.append({"cve_id": cve_id, "language": language, "reason": "duplicate_or_outside_test"})
            continue
        if row.get("original_text") != records[cve_id].description:
            excluded_pairs.append({"cve_id": cve_id, "language": language, "reason": "source_mismatch"})
            continue
        if row.get("severity") and row["severity"] != records[cve_id].severity:
            excluded_pairs.append({"cve_id": cve_id, "language": language, "reason": "severity_mismatch"})
            continue
        validation_status, _ = validate_translation_for_language(
            records[cve_id].description, row.get("translated_text", ""), language
        )
        if validation_status != "valid":
            excluded_pairs.append(
                {"cve_id": cve_id, "language": language, "reason": validation_status}
            )
            continue
        if row.get("validation_status") not in {"", "valid"}:
            validation_status_disagreements.append(
                {
                    "cve_id": cve_id,
                    "language": language,
                    "stored_status": row.get("validation_status", ""),
                    "recomputed_status": validation_status,
                }
            )
        translation_by_pair[pair] = row

    comparison_predictions: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    for comparison_dir in args.comparison_prediction_dir:
        if not comparison_dir.is_dir():
            raise FileNotFoundError(
                f"Comparison prediction directory not found: {comparison_dir}"
            )
        model_name = (
            comparison_dir.parent.parent.name
            if comparison_dir.name.lower() == "predictions"
            else comparison_dir.name
        )
        if model_name in comparison_predictions:
            raise ValueError(f"Duplicate comparison model name: {model_name}")
        comparison_predictions[model_name] = {}
        for language in LANGUAGES:
            prediction_path = comparison_dir / f"{language}_test_predictions.csv"
            if not prediction_path.is_file():
                raise FileNotFoundError(
                    f"Missing {language} comparison predictions: {prediction_path}"
                )
            comparison_predictions[model_name][language] = read_prediction_csv(
                prediction_path
            )

    baseline_english = read_prediction_csv(args.baseline_english)
    baseline_translations: dict[tuple[str, str], dict[str, str]] = {}
    with args.baseline_translations.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            pair = (row["cve_id"], row["language"].lower())
            if pair in baseline_translations:
                raise ValueError(f"Duplicate baseline translation prediction: {pair}")
            baseline_translations[pair] = row

    predictions_by_language: dict[str, dict[str, dict[str, Any]]] = {}
    baseline_by_language: dict[str, dict[str, dict[str, Any]]] = {}
    language_metrics: dict[str, Any] = {}
    baseline_metrics: dict[str, Any] = {}
    metrics_dir = output / "metrics"
    metrics_dir.mkdir()

    for language in LANGUAGES:
        language_ids = test_ids if language == "en" else [
            cve_id for cve_id in test_ids if (cve_id, language) in translation_by_pair
        ]
        texts = [
            records[cve_id].description
            if language == "en"
            else translation_by_pair[(cve_id, language)]["translated_text"]
            for cve_id in language_ids
        ]
        predicted, probabilities = predict_texts(
            model, tokenizer, texts, device, args.batch_size, args.max_length
        )
        rows = [
            {
                "cve_id": cve_id,
                "language": language,
                "true_label": records[cve_id].severity,
                "predicted_label": prediction,
                **{
                    f"probability_{label}": float(probabilities[offset][index])
                    for index, label in enumerate(LABELS)
                },
            }
            for offset, (cve_id, prediction) in enumerate(zip(language_ids, predicted))
        ]
        predictions_by_language[language] = {row["cve_id"]: row for row in rows}
        pred_path = write_prediction_csv(output, language, rows)
        language_metrics[language] = metrics_for_rows(rows, args.seed)
        language_metrics[language]["test_cve_ids"] = language_ids
        language_metrics[language]["prediction_path"] = str(pred_path)
        (metrics_dir / f"{language}_metrics.json").write_text(
            json.dumps(language_metrics[language], indent=2), encoding="utf-8"
        )

        if language == "en":
            baseline_rows = [
                {
                    "cve_id": cve_id,
                    "true_label": records[cve_id].severity,
                    "predicted_label": baseline_english[cve_id]["predicted_label"],
                }
                for cve_id in test_ids
            ]
        else:
            baseline_rows = [
                {
                    "cve_id": cve_id,
                    "true_label": records[cve_id].severity,
                    "predicted_label": baseline_translations[(cve_id, language)][
                        "translated_predicted_label"
                    ],
                }
                for cve_id in language_ids
            ]
        baseline_by_language[language] = {row["cve_id"]: row for row in baseline_rows}
        baseline_metrics[language] = metrics_for_rows(baseline_rows, args.seed)
        comparison_ids = language_ids
        language_metrics[language]["baseline_comparison"] = {
            "baseline_sample_count": len(baseline_rows),
            "same_cve_set": len(baseline_rows) == len(comparison_ids)
            and set(row["cve_id"] for row in baseline_rows) == set(comparison_ids),
            "baseline_macro_f1": baseline_metrics[language]["macro_f1"],
            "xlmr_minus_baseline_macro_f1": (
                language_metrics[language]["macro_f1"] - baseline_metrics[language]["macro_f1"]
            ),
            "paired_macro_f1_difference_95_ci": paired_bootstrap_macro_f1_difference(
                [row["true_label"] for row in rows],
                [row["predicted_label"] for row in rows],
                [baseline_by_language[language][cve_id]["predicted_label"] for cve_id in comparison_ids],
                args.seed,
            ),
        }
        (metrics_dir / f"{language}_metrics.json").write_text(
            json.dumps(language_metrics[language], indent=2), encoding="utf-8"
        )

    paired_ids = set.intersection(
        *(set(predictions_by_language[language]) for language in LANGUAGES)
    )
    cross_language: dict[str, Any] = {
        "aligned_cve_count": len(paired_ids),
        "aligned_cve_ids": sorted(paired_ids),
        "macro_f1_difference_from_english": {},
        "agreement_with_english": {},
        "severity_flip_rate_from_english": {},
        "critical_risk_disagreement_with_english": {},
        "pairwise_prediction_agreement": {},
    }
    english_aligned = [predictions_by_language["en"][cve_id] for cve_id in sorted(paired_ids)]
    for language in TARGET_LANGUAGES:
        aligned_truth = [row["true_label"] for row in english_aligned]
        english_predictions = [row["predicted_label"] for row in english_aligned]
        target_predictions = [
            predictions_by_language[language][cve_id]["predicted_label"]
            for cve_id in sorted(paired_ids)
        ]
        english_macro = f1_score(
            aligned_truth, english_predictions, labels=LABELS, average="macro", zero_division=0
        )
        target_macro = f1_score(
            aligned_truth, target_predictions, labels=LABELS, average="macro", zero_division=0
        )
        agreement = agreement_metrics(english_predictions, target_predictions)
        cross_language["macro_f1_difference_from_english"][language] = {
            "english_macro_f1_on_aligned_cves": float(english_macro),
            "target_macro_f1_on_aligned_cves": float(target_macro),
            "target_minus_english": float(target_macro - english_macro),
            "paired_95_ci": paired_bootstrap_macro_f1_difference(
                aligned_truth, target_predictions, english_predictions, args.seed
            ),
        }
        cross_language["agreement_with_english"][language] = agreement
        cross_language["severity_flip_rate_from_english"][language] = {
            "flip_count": agreement["disagreement_count"],
            "flip_rate": 1.0 - agreement["agreement_with_english"],
        }
        critical_disagreements = sum(
            (left == "Critical") != (right == "Critical")
            for left, right in zip(english_predictions, target_predictions)
        )
        cross_language["critical_risk_disagreement_with_english"][language] = {
            "disagreement_count": critical_disagreements,
            "disagreement_rate": critical_disagreements / len(paired_ids) if paired_ids else None,
        }
    for left_index, left in enumerate(LANGUAGES):
        for right in LANGUAGES[left_index + 1 :]:
            left_predictions = [
                predictions_by_language[left][cve_id]["predicted_label"]
                for cve_id in sorted(paired_ids)
            ]
            right_predictions = [
                predictions_by_language[right][cve_id]["predicted_label"]
                for cve_id in sorted(paired_ids)
            ]
            cross_language["pairwise_prediction_agreement"][f"{left}:{right}"] = agreement_metrics(
                left_predictions, right_predictions
            )

    baseline_summary = {}
    for language in LANGUAGES:
        baseline_rows = list(baseline_by_language[language].values())
        baseline_summary[language] = metrics_for_rows(baseline_rows, args.seed)
    additional_comparison_metrics: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name, comparison_model_predictions in comparison_predictions.items():
        additional_comparison_metrics[model_name] = {}
        for language in LANGUAGES:
            current_rows = list(comparison_model_predictions[language].values())
            current_ids = {row["cve_id"] for row in current_rows}
            expected_ids = set(language_metrics[language]["test_cve_ids"])
            if not expected_ids.issubset(current_ids):
                raise ValueError(
                    f"{model_name} {language} predictions miss fixed valid test CVEs: "
                    f"{sorted(expected_ids - current_ids)[:5]}"
                )
            rows = []
            for cve_id in language_metrics[language]["test_cve_ids"]:
                prior = comparison_model_predictions[language][cve_id]
                expected_label = records[cve_id].severity
                if prior.get("true_label") != expected_label:
                    raise ValueError(
                        f"{model_name} {language} ground-truth label mismatch for {cve_id}"
                    )
                rows.append(
                    {
                        "cve_id": cve_id,
                        "true_label": expected_label,
                        "predicted_label": prior["predicted_label"],
                    }
                )
            additional_comparison_metrics[model_name][language] = metrics_for_rows(
                rows, args.seed
            )
            additional_comparison_metrics[model_name][language]["test_cve_ids"] = [
                row["cve_id"] for row in rows
            ]
    comparison_model_metrics = {
        "tfidf_logistic_regression": baseline_summary,
        experiment_name: language_metrics,
        **additional_comparison_metrics,
    }
    comparison_differences: dict[str, dict[str, Any]] = {}
    for model_name in comparison_model_metrics:
        if model_name == experiment_name:
            continue
        comparison_differences[model_name] = {}
        for language in LANGUAGES:
            cve_ids = language_metrics[language]["test_cve_ids"]
            truth = [records[cve_id].severity for cve_id in cve_ids]
            current_predictions = [
                predictions_by_language[language][cve_id]["predicted_label"]
                for cve_id in cve_ids
            ]
            if model_name == "tfidf_logistic_regression":
                prior_predictions = [
                    baseline_by_language[language][cve_id]["predicted_label"]
                    for cve_id in cve_ids
                ]
            else:
                prior_predictions = [
                    comparison_predictions[model_name][language][cve_id][
                        "predicted_label"
                    ]
                    for cve_id in cve_ids
                ]
            comparison_differences[model_name][language] = paired_model_difference_metrics(
                truth,
                current_predictions,
                prior_predictions,
                language_metrics[language],
                comparison_model_metrics[model_name][language],
                args.seed,
            )
    summary = {
        "status": "complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment_name,
        "checkpoint_path": str(checkpoint),
        "sample_counts": {language: language_metrics[language]["sample_count"] for language in LANGUAGES},
        "excluded_rows": excluded_pairs,
        "excluded_counts": dict(Counter(row["reason"] for row in excluded_pairs)),
        "paired_cross_language_cve_count": len(paired_ids),
        "translation_human_review": {
            "performed": False,
            "status": "not_human_reviewed",
        },
        "classifier_evaluation_status": "completed",
        "test_results_interpretation": (
            "exploratory_prior_test_results_previously_observed"
            if args.exploratory_test_comparison
            else "test_set_evaluation"
        ),
        "comparison_models": [
            "tfidf_logistic_regression",
            experiment_name,
            *additional_comparison_metrics.keys(),
        ],
        "evaluation_runtime_seconds": time.perf_counter() - started,
    }
    provenance = {
        "status": "completed",
        "experiment": experiment_name,
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "created_at": summary["created_at"],
        "records_path": str(args.records.resolve()),
        "records_sha256": sha256_file(args.records),
        "splits_path": str(args.splits.resolve()),
        "splits_sha256": sha256_file(args.splits),
        "translations_path": str(args.translations.resolve()),
        "translations_sha256": sha256_file(args.translations),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": {
            path.name: sha256_file(path) for path in checkpoint.iterdir() if path.is_file()
        },
        "label_order": LABELS,
        "ordinal_order": ORDINAL_ORDER,
        "translation_validation_status_disagreements": validation_status_disagreements,
        "excluded_pairs": excluded_pairs,
        "software": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "scikit-learn": package_version("scikit-learn"),
            "numpy": np.__version__,
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "cuda_version": torch.version.cuda,
        },
        "evaluation_config": {
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": args.seed,
        },
        "comparison_prediction_dirs": [
            str(path.resolve()) for path in args.comparison_prediction_dir
        ],
        "test_results_interpretation": summary["test_results_interpretation"],
    }
    (output / "per_language_metrics.json").write_text(
        json.dumps(language_metrics, indent=2), encoding="utf-8"
    )
    (output / "cross_language_metrics.json").write_text(
        json.dumps(cross_language, indent=2), encoding="utf-8"
    )
    (output / "baseline_comparison.json").write_text(
        json.dumps(
            {
                "status": "paired_comparison",
                "same_cve_ids_per_language": True,
                "models": comparison_model_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "comparison_differences.json").write_text(
        json.dumps(comparison_differences, indent=2), encoding="utf-8"
    )
    (output / "excluded_rows.json").write_text(
        json.dumps(
            {
                "excluded_pairs": excluded_pairs,
                "counts": dict(Counter(row["reason"] for row in excluded_pairs)),
                "validation_status_disagreements": validation_status_disagreements,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    comparison_path = output / "comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "language",
                "model",
                "sample_count",
                "macro_f1",
                "macro_f1_95_ci_low",
                "macro_f1_95_ci_high",
                "critical_false_negative_count",
                "accuracy",
                "weighted_f1",
                "mcc",
            ],
        )
        writer.writeheader()
        writer.writerows(
            comparison_rows(comparison_model_metrics)
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
