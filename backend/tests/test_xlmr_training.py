import csv
import hashlib

import numpy as np
import pytest

from scripts.train_xlmr import (
    inverse_sqrt_class_weights,
    loss_is_scaled_for_gradient_accumulation,
    prepare_multilingual_examples,
)
from scripts.evaluate_xlmr_research import (
    LANGUAGES,
    comparison_rows,
    paired_model_difference_metrics,
    write_prediction_csv,
)
from scripts.translate_expanded_test import (
    _read_pair_manifest,
    _write_rows,
    ids_for_split_scope,
)
from scripts.tune_critical_threshold_validation import (
    critical_threshold_predictions,
    select_shared_threshold,
)


def test_inverse_sqrt_class_weights_are_normalized_and_follow_label_order() -> None:
    weights = inverse_sqrt_class_weights([198, 524, 77, 715])

    assert len(weights) == 4
    assert sum(weights) / len(weights) == pytest.approx(1.0)
    assert weights[2] > weights[0] > weights[1] > weights[3]


def test_inverse_sqrt_class_weights_reject_missing_training_classes() -> None:
    with pytest.raises(ValueError, match="Every severity class"):
        inverse_sqrt_class_weights([10, 5, 0, 2])


def test_custom_weighted_loss_is_scaled_by_trainer_for_gradient_accumulation() -> None:
    assert loss_is_scaled_for_gradient_accumulation("inverse_sqrt") is False
    assert loss_is_scaled_for_gradient_accumulation("none") is None


def test_comparison_rows_identify_the_actual_xlmr_experiment() -> None:
    metric = {
        "sample_count": 1,
        "macro_f1": 0.5,
        "bootstrap_95_ci": {"macro_f1_95_ci": [0.4, 0.6]},
        "critical_class": {"false_negative_count": 0},
        "accuracy": 0.5,
        "weighted_f1": 0.5,
        "matthews_correlation_coefficient": 0.0,
    }
    summaries = {language: metric for language in LANGUAGES}

    rows = comparison_rows(
        {
            "tfidf_logistic_regression": summaries,
            "xlmr_multilingual_test": summaries,
            "xlmr_english_inverse_sqrt_test": summaries,
        }
    )

    model_names = {row["model"] for row in rows}
    assert model_names == {
        "tfidf_logistic_regression",
        "xlmr_multilingual_test",
        "xlmr_english_inverse_sqrt_test",
    }


def test_prediction_writer_does_not_overwrite_comparison_predictions(tmp_path) -> None:
    output_dir = tmp_path / "multilingual-evaluation"
    output_predictions = output_dir / "predictions"
    output_predictions.mkdir(parents=True)
    comparison_predictions = tmp_path / "weighted-model" / "predictions"
    comparison_predictions.mkdir(parents=True)
    comparison_file = comparison_predictions / "en_test_predictions.csv"
    comparison_file.write_text("preserve this comparator\n", encoding="utf-8")

    output_path = write_prediction_csv(
        output_dir,
        "en",
        [{"cve_id": "test-cve", "predicted_label": "High"}],
    )

    assert output_path == output_predictions / "en_test_predictions.csv"
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "cve_id,predicted_label",
        "test-cve,High",
    ]
    assert comparison_file.read_text(encoding="utf-8") == "preserve this comparator\n"


def test_paired_model_difference_is_current_minus_comparator() -> None:
    current_metrics = {
        "macro_f1": 0.25,
        "critical_class": {"false_negative_count": 1},
    }
    comparator_metrics = {
        "macro_f1": 0.0,
        "critical_class": {"false_negative_count": 3},
    }

    result = paired_model_difference_metrics(
        ["High"] * 4,
        ["High"] * 4,
        ["Critical"] * 4,
        current_metrics,
        comparator_metrics,
        seed=42,
    )

    assert result["macro_f1_difference_current_minus_comparator"] == pytest.approx(0.25)
    assert result["paired_macro_f1_difference_95_ci"] == pytest.approx([0.25, 0.25])
    assert result["critical_false_negative_difference_current_minus_comparator"] == -2


def test_shared_critical_threshold_overrides_argmax_and_resolves_conflicts() -> None:
    probabilities = [
        [0.40, 0.35, 0.15, 0.10],
        [0.40, 0.10, 0.45, 0.05],
    ]

    assert critical_threshold_predictions(np.array(probabilities), 0.35) == [
        "Critical",
        "Critical",
    ]
    assert critical_threshold_predictions(np.array(probabilities), 0.41) == [
        "High",
        "Low",
    ]


def test_threshold_selection_requires_macro_gain_critical_recall_gain_and_no_low_regression() -> None:
    baseline = {
        "mean_macro_f1": 0.5,
        "mean_critical_recall": 0.6,
        "per_language": {
            language: {"low_recall": 0.3} for language in LANGUAGES
        },
    }
    candidates = {
        "0.40": {
            "mean_macro_f1": 0.55,
            "mean_critical_recall": 0.65,
            "per_language": {
                language: {"low_recall": 0.3} for language in LANGUAGES
            },
        },
        "0.45": {
            "mean_macro_f1": 0.60,
            "mean_critical_recall": 0.7,
            "per_language": {
                language: {"low_recall": 0.3} for language in LANGUAGES
            },
        },
        "0.50": {
            "mean_macro_f1": 0.7,
            "mean_critical_recall": 0.8,
            "per_language": {
                language: {
                    "low_recall": 0.2 if language == "fr" else 0.3
                }
                for language in LANGUAGES
            },
        },
    }

    assert select_shared_threshold(baseline, candidates) == pytest.approx(0.45)

    candidates["0.40"]["mean_macro_f1"] = 0.49
    candidates["0.45"]["mean_critical_recall"] = 0.59
    assert select_shared_threshold(baseline, candidates) is None


def test_train_validation_scope_never_selects_test_cves() -> None:
    splits = {"train": ["train-cve"], "validation": ["validation-cve"], "test": ["test-cve"]}

    assert ids_for_split_scope(splits, "train_validation") == ["train-cve", "validation-cve"]


def test_pair_manifest_reader_accepts_power_shell_utf8_bom(tmp_path) -> None:
    path = tmp_path / "pairs.csv"
    path.write_text("\ufeffcve_id,language\ntrain-cve,hi\n", encoding="utf-8")

    assert _read_pair_manifest(path) == {("train-cve", "hi")}


def test_resumable_translation_checkpoint_retries_transient_file_lock(
    tmp_path, monkeypatch
) -> None:
    import scripts.translate_expanded_test as translation

    original_replace = translation.os.replace
    attempts = 0

    def replace_with_transient_lock(source, destination):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("transient file lock")
        original_replace(source, destination)

    monkeypatch.setattr(translation.os, "replace", replace_with_transient_lock)
    output = tmp_path / "translations.csv"

    _write_rows(
        output,
        [{"cve_id": "train-cve", "language": "hi", "translated_text": "text"}],
    )

    assert attempts == 2
    assert _read_pair_manifest(output) == {("train-cve", "hi")}
    assert output.read_text(encoding="utf-8").startswith("cve_id,language,severity")


def test_multilingual_examples_exclude_test_cves_and_record_missing_pairs(
    tmp_path, monkeypatch
) -> None:
    import scripts.train_xlmr as training

    records = {
        cve_id: {"description": f"description for {cve_id}", "severity": "High"}
        for cve_id in ("train-cve", "validation-cve", "test-cve")
    }
    train_source = records["train-cve"]["description"]
    rows = [
        {
            "cve_id": "train-cve",
            "language": "hi",
            "severity": "High",
            "source_sha256": hashlib.sha256(train_source.encode("utf-8")).hexdigest(),
            "original_text": train_source,
            "translated_text": "validated Hindi text",
            "validation_status": "valid",
        },
        {
            "cve_id": "test-cve",
            "language": "hi",
            "severity": "High",
            "source_sha256": hashlib.sha256(
                records["test-cve"]["description"].encode("utf-8")
            ).hexdigest(),
            "original_text": records["test-cve"]["description"],
            "translated_text": "test translation must not enter training",
            "validation_status": "valid",
        },
        {
            "cve_id": "validation-cve",
            "language": "es",
            "severity": "High",
            "source_sha256": hashlib.sha256(
                records["validation-cve"]["description"].encode("utf-8")
            ).hexdigest(),
            "original_text": "mismatched source text",
            "translated_text": "must not be admitted",
            "validation_status": "valid",
        },
        {
            "cve_id": "validation-cve",
            "language": "ar",
            "severity": "High",
            "source_sha256": hashlib.sha256(
                records["validation-cve"]["description"].encode("utf-8")
            ).hexdigest(),
            "original_text": records["validation-cve"]["description"],
            "translated_text": "",
            "validation_status": "failed",
        },
        {
            "cve_id": "validation-cve",
            "language": "zh",
            "severity": "High",
            "source_sha256": hashlib.sha256(
                records["validation-cve"]["description"].encode("utf-8")
            ).hexdigest(),
            "original_text": records["validation-cve"]["description"],
            "translated_text": "truncated output",
            "validation_status": "failed",
            "output_truncated": "true",
            "incomplete_units": "1",
        },
    ]
    translations_path = tmp_path / "translations.csv"
    with translations_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    monkeypatch.setattr(
        training,
        "validate_translation_for_language",
        lambda original, translated, language: ("valid", []),
    )

    train_examples, validation_examples, report = prepare_multilingual_examples(
        records,
        ["train-cve"],
        ["validation-cve"],
        translations_path,
    )

    assert [(item["cve_id"], item["language"]) for item in train_examples] == [
        ("train-cve", "en"),
        ("train-cve", "hi"),
    ]
    assert [(item["cve_id"], item["language"]) for item in validation_examples] == [
        ("validation-cve", "en")
    ]
    assert report["test_cves_used"] == 0
    assert report["counts_by_exclusion_reason"]["outside_train_validation_split"] == 1
    assert report["counts_by_exclusion_reason"]["source_text_mismatch"] == 1
    assert report["counts_by_exclusion_reason"]["empty_translation"] == 1
    assert report["counts_by_exclusion_reason"]["missing_translation_pair"] == 6
    assert report["row_diagnostics"]["empty_translation_count"] == 1
    assert report["row_diagnostics"]["output_truncated_count"] == 1
