"""Train XLM-R using only the fixed manifest's train and validation CVEs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.translations import TRANSLATION_LANGUAGES, validate_translation_for_language

LABELS = ["Critical", "High", "Low", "Medium"]
DEFAULT_MODEL = "FacebookAI/xlm-roberta-base"
DEFAULT_REVISION = "e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
LANGUAGES = ("en", *TRANSLATION_LANGUAGES)


def inverse_sqrt_class_weights(counts: list[int]) -> list[float]:
    if not counts or any(count <= 0 for count in counts):
        raise ValueError("Every severity class must have at least one training example")
    weights = [count ** -0.5 for count in counts]
    mean_weight = sum(weights) / len(weights)
    return [weight / mean_weight for weight in weights]


def loss_is_scaled_for_gradient_accumulation(class_weighting: str) -> bool | None:
    if class_weighting == "inverse_sqrt":
        return False
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ids(ids: list[str]) -> str:
    payload = json.dumps(sorted(ids), separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def prepare_multilingual_examples(
    records_by_id: dict[str, dict[str, Any]],
    train_ids: list[str],
    validation_ids: list[str],
    translations_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train_set = set(train_ids)
    validation_set = set(validation_ids)
    allowed_ids = train_set | validation_set
    split_for_id = {cve_id: "train" for cve_id in train_ids}
    split_for_id.update({cve_id: "validation" for cve_id in validation_ids})
    examples_by_split: dict[str, list[dict[str, Any]]] = {
        "train": [
            {
                "cve_id": cve_id,
                "language": "en",
                "description": records_by_id[cve_id]["description"],
                "severity": records_by_id[cve_id]["severity"],
            }
            for cve_id in train_ids
        ],
        "validation": [
            {
                "cve_id": cve_id,
                "language": "en",
                "description": records_by_id[cve_id]["description"],
                "severity": records_by_id[cve_id]["severity"],
            }
            for cve_id in validation_ids
        ],
    }
    with translations_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {
            "cve_id",
            "language",
            "severity",
            "source_sha256",
            "original_text",
            "translated_text",
            "validation_status",
        }
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(
                f"Translation CSV is missing required columns: {sorted(required - set(reader.fieldnames or []))}"
            )
        rows = list(reader)

    pair_counts = Counter(
        (
            row.get("cve_id", "").strip(),
            row.get("language", "").strip().lower(),
        )
        for row in rows
    )
    accepted_pairs: set[tuple[str, str]] = set()
    excluded: list[dict[str, str]] = []
    for row in rows:
        cve_id = row.get("cve_id", "").strip()
        language = row.get("language", "").strip().lower()
        pair = (cve_id, language)
        reason = ""
        if pair_counts[pair] != 1:
            reason = "duplicate_pair"
        elif cve_id not in allowed_ids:
            reason = "outside_train_validation_split"
        elif language not in TRANSLATION_LANGUAGES:
            reason = "unsupported_language"
        elif row.get("original_text") != records_by_id[cve_id]["description"]:
            reason = "source_text_mismatch"
        elif row.get("severity") != records_by_id[cve_id]["severity"]:
            reason = "severity_mismatch"
        elif row.get("source_sha256") != hashlib.sha256(
            records_by_id[cve_id]["description"].encode("utf-8")
        ).hexdigest():
            reason = "source_hash_mismatch"
        elif not row.get("translated_text", "").strip():
            reason = "empty_translation"
        elif row.get("validation_status") != "valid":
            reason = f"pipeline_status:{row.get('validation_status') or 'missing'}"
        else:
            recomputed_status, _ = validate_translation_for_language(
                records_by_id[cve_id]["description"],
                row["translated_text"],
                language,
            )
            if recomputed_status != "valid":
                reason = f"recomputed_status:{recomputed_status}"
            elif row.get("source_truncated", "").lower() == "true":
                reason = "source_truncated"
            elif row.get("output_truncated", "").lower() == "true":
                reason = "output_truncated"
            elif row.get("incomplete_units", "0") not in {"", "0"}:
                reason = "incomplete_translation_units"
            elif row.get("missing_identifiers", "").strip():
                reason = "missing_technical_identifiers"
            elif row.get("missing_placeholders", "0") not in {"", "0"}:
                reason = "missing_technical_placeholders"
            elif row.get("placeholder_error"):
                reason = "placeholder_restore_error"
            elif row.get("identifier_integrity_error"):
                reason = "technical_identifier_integrity_error"
            elif row.get("error") or row.get("validation_error") or row.get(
                "identifier_integrity_error"
            ):
                reason = "recorded_translation_error"
        if reason:
            excluded.append(
                {
                    "cve_id": cve_id,
                    "language": language,
                    "split": split_for_id.get(cve_id, "outside"),
                    "reason": reason,
                    "validation_status": row.get("validation_status", ""),
                    "source_sha256": row.get("source_sha256", ""),
                    "source_truncated": row.get("source_truncated", ""),
                    "output_truncated": row.get("output_truncated", ""),
                    "incomplete_units": row.get("incomplete_units", ""),
                    "missing_identifiers": row.get("missing_identifiers", ""),
                    "missing_placeholders": row.get("missing_placeholders", ""),
                    "placeholder_error": row.get("placeholder_error", ""),
                    "translated_text_empty": str(
                        not bool(row.get("translated_text", "").strip())
                    ).lower(),
                    "error": row.get("error", ""),
                    "validation_error": row.get("validation_error", ""),
                    "identifier_integrity_error": row.get(
                        "identifier_integrity_error", ""
                    ),
                }
            )
            continue
        accepted_pairs.add(pair)
        examples_by_split[split_for_id[cve_id]].append(
            {
                "cve_id": cve_id,
                "language": language,
                "description": row["translated_text"],
                "severity": records_by_id[cve_id]["severity"],
            }
        )

    missing = [
        {
            "cve_id": cve_id,
            "language": language,
            "split": split_for_id[cve_id],
            "reason": "missing_translation_pair",
        }
        for cve_id in train_ids + validation_ids
        for language in TRANSLATION_LANGUAGES
        if (cve_id, language) not in pair_counts
    ]

    def summarize(examples: list[dict[str, Any]]) -> dict[str, Any]:
        language_counts = Counter(example["language"] for example in examples)
        label_counts = Counter(example["severity"] for example in examples)
        return {
            "example_count": len(examples),
            "counts_by_language": {
                language: language_counts[language] for language in LANGUAGES
            },
            "label_counts": {label: label_counts[label] for label in LABELS},
            "label_counts_by_language": {
                language: {
                    label: sum(
                        example["language"] == language
                        and example["severity"] == label
                        for example in examples
                    )
                    for label in LABELS
                }
                for language in LANGUAGES
            },
        }

    diagnostics = {
        "row_count": len(rows),
        "rows_by_validation_status": dict(
            Counter(row.get("validation_status", "") for row in rows)
        ),
        "empty_translation_count": sum(
            not row.get("translated_text", "").strip() for row in rows
        ),
        "source_truncated_count": sum(
            row.get("source_truncated", "").lower() == "true" for row in rows
        ),
        "output_truncated_count": sum(
            row.get("output_truncated", "").lower() == "true" for row in rows
        ),
        "incomplete_translation_count": sum(
            row.get("incomplete_units", "0") not in {"", "0"} for row in rows
        ),
        "missing_identifier_count": sum(
            bool(row.get("missing_identifiers", "").strip()) for row in rows
        ),
        "missing_placeholder_count": sum(
            row.get("missing_placeholders", "0") not in {"", "0"} for row in rows
        ),
        "recorded_error_count": sum(
            bool(
                row.get("error")
                or row.get("validation_error")
                or row.get("identifier_integrity_error")
                or row.get("placeholder_error")
            )
            for row in rows
        ),
    }
    report = {
        "translation_file": str(translations_path.resolve()),
        "translation_file_sha256": sha256_file(translations_path),
        "expected_target_translation_pairs": len(allowed_ids) * len(TRANSLATION_LANGUAGES),
        "valid_target_translation_pairs": len(accepted_pairs),
        "excluded_rows": excluded,
        "missing_pairs": missing,
        "row_diagnostics": diagnostics,
        "counts_by_exclusion_reason": dict(
            Counter(row["reason"] for row in excluded + missing)
        ),
        "split_summaries": {
            split_name: summarize(examples)
            for split_name, examples in examples_by_split.items()
        },
        "missing_languages_by_split": {
            split_name: [
                language
                for language in LANGUAGES
                if not any(
                    example["language"] == language
                    for example in examples
                )
            ]
            for split_name, examples in examples_by_split.items()
        },
        "test_cves_used": 0,
        "translation_human_review": "not_performed",
    }
    return examples_by_split["train"], examples_by_split["validation"], report


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune XLM-R using English or validated multilingual fixed-split data."
    )
    parser.add_argument("--data", type=Path, default=Path("data/expanded/nvd_records.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/expanded/cve_splits.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--multilingual-translations",
        type=Path,
        default=None,
        help="Automated-validation translation CSV for fixed train/validation CVEs.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stopping-patience", type=int, default=1)
    parser.add_argument(
        "--class-weighting",
        choices=("none", "inverse_sqrt"),
        default="none",
        help="Optional severity weighting calculated only from fixed-split training examples.",
    )
    args = parser.parse_args()

    import numpy as np
    import torch
    from sklearn.metrics import f1_score, precision_recall_fscore_support
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this XLM-R research run.")
    if args.epochs < 1 or args.batch_size < 1 or args.max_length < 1:
        raise ValueError("Epoch, batch-size, and max-length settings must be positive.")

    records = json.loads(args.data.read_text(encoding="utf-8"))
    records_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        cve_id = record["cve_id"]
        if cve_id in records_by_id:
            raise ValueError(f"Duplicate CVE ID in dataset: {cve_id}")
        records_by_id[cve_id] = record
    split = json.loads(args.splits.read_text(encoding="utf-8"))["splits"]
    train_ids = split["train"]
    validation_ids = split["validation"]
    test_ids = split["test"]
    for name, ids in (("train", train_ids), ("validation", validation_ids), ("test", test_ids)):
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate CVE IDs in {name} split")
        missing = set(ids) - records_by_id.keys()
        if missing:
            raise ValueError(f"{name} split contains IDs missing from dataset: {sorted(missing)[:5]}")
    if set(train_ids) & set(validation_ids) or set(train_ids) & set(test_ids) or set(validation_ids) & set(test_ids):
        raise ValueError("Fixed split has overlapping CVE IDs")

    train_records = [records_by_id[cve_id] for cve_id in train_ids]
    validation_records = [records_by_id[cve_id] for cve_id in validation_ids]
    if any(record["severity"] not in LABELS for record in train_records + validation_records):
        raise ValueError("Train/validation data contains a severity outside the fixed label mapping")
    translation_report = None
    if args.multilingual_translations is not None:
        train_examples, validation_examples, translation_report = prepare_multilingual_examples(
            records_by_id, train_ids, validation_ids, args.multilingual_translations
        )
    else:
        train_examples = [
            {
                "cve_id": cve_id,
                "language": "en",
                "description": records_by_id[cve_id]["description"],
                "severity": records_by_id[cve_id]["severity"],
            }
            for cve_id in train_ids
        ]
        validation_examples = [
            {
                "cve_id": cve_id,
                "language": "en",
                "description": records_by_id[cve_id]["description"],
                "severity": records_by_id[cve_id]["severity"],
            }
            for cve_id in validation_ids
        ]
    label_to_id = {label: index for index, label in enumerate(LABELS)}
    train_class_counts = Counter(example["severity"] for example in train_examples)
    class_weights = (
        inverse_sqrt_class_weights([train_class_counts[label] for label in LABELS])
        if args.class_weighting == "inverse_sqrt"
        else None
    )
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty experiment directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    device = torch.cuda.get_device_properties(0)
    started_at = datetime.now(timezone.utc).isoformat()
    manifest_path = output / "training_provenance.json"
    config = {
        "model": args.model,
        "revision": args.revision,
        "tokenizer_revision": args.revision,
        "epochs": args.epochs,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "seed": args.seed,
        "early_stopping_patience": args.early_stopping_patience,
        "selection_metric": "eval_macro_f1",
        "selection_direction": "maximize",
        "precision": "fp16",
        "gradient_checkpointing": True,
        "class_weighting": args.class_weighting,
        "class_weight_source": "training_split_only" if class_weights is not None else None,
        "class_weights": (
            {label: class_weights[index] for index, label in enumerate(LABELS)}
            if class_weights is not None
            else None
        ),
        "training_class_counts": {label: train_class_counts[label] for label in LABELS},
        "training_languages": list(LANGUAGES)
        if args.multilingual_translations is not None
        else ["en"],
    }
    manifest: dict[str, Any] = {
        "status": "training",
        "experiment": output.name,
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "started_at": started_at,
        "data_path": str(args.data.resolve()),
        "split_path": str(args.splits.resolve()),
        "dataset_sha256": sha256_file(args.data),
        "split_sha256": sha256_file(args.splits),
        "model_initialization": args.model,
        "model_revision": args.revision,
        "tokenizer_revision": args.revision,
        "label_mapping": label_to_id,
        "label_order": LABELS,
        "split_counts": {
            "train": len(train_ids),
            "validation": len(validation_ids),
            "test": len(test_ids),
        },
        "split_id_sha256": {
            "train": sha256_ids(train_ids),
            "validation": sha256_ids(validation_ids),
            "test": sha256_ids(test_ids),
        },
        "split_cve_ids": {
            "train": train_ids,
            "validation": validation_ids,
            "test": test_ids,
        },
        "test_ids_used_for_training_or_selection": False,
        "training_data_mode": "multilingual" if translation_report else "english",
        "multilingual_translation_report": (
            "translation_validation_report.json" if translation_report else None
        ),
        "multilingual_translations_path": (
            str(args.multilingual_translations.resolve())
            if args.multilingual_translations is not None
            else None
        ),
        "multilingual_translations_sha256": (
            sha256_file(args.multilingual_translations)
            if args.multilingual_translations is not None
            else None
        ),
        "training_example_counts": {
            "train": len(train_examples),
            "validation": len(validation_examples),
        },
        "training_validation_language_counts": (
            {
                "train": translation_report["split_summaries"]["train"][
                    "counts_by_language"
                ],
                "validation": translation_report["split_summaries"]["validation"][
                    "counts_by_language"
                ],
            }
            if translation_report
            else {
                "train": {"en": len(train_examples)},
                "validation": {"en": len(validation_examples)},
            }
        ),
        "config": config,
        "hardware": {
            "device": device.name,
            "total_gpu_memory_bytes": device.total_memory,
            "cuda_version": torch.version.cuda,
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
            "scikit-learn": package_version("scikit-learn"),
            "numpy": np.__version__,
        },
        "human_translation_review": (
            "machine_translations_not_human_reviewed"
            if translation_report
            else "not_applicable_to_english_training_data"
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if translation_report is not None:
        (output / "translation_validation_report.json").write_text(
            json.dumps(translation_report, indent=2), encoding="utf-8"
        )
        missing_languages = translation_report["missing_languages_by_split"]
        if any(missing_languages.values()):
            manifest.update(
                {
                    "status": "blocked_incomplete_language_coverage",
                    "blocking_missing_languages_by_split": missing_languages,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
            )
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(
                json.dumps(
                    {
                        "status": manifest["status"],
                        "missing_languages_by_split": missing_languages,
                        "translation_report": str(
                            output / "translation_validation_report.json"
                        ),
                    },
                    indent=2,
                )
            )
            return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        revision=args.revision,
        num_labels=len(LABELS),
        id2label={index: label for label, index in label_to_id.items()},
        label2id=label_to_id,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    class VulnerabilityDataset(Dataset):
        def __init__(self, examples: list[dict[str, Any]]) -> None:
            self.examples = examples
            self.tokens = tokenizer(
                [example["description"] for example in examples],
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
            )

        def __len__(self) -> int:
            return len(self.examples)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            return {
                key: torch.tensor(value[index], dtype=torch.long)
                for key, value in self.tokens.items()
            } | {
                "labels": torch.tensor(
                    label_to_id[self.examples[index]["severity"]], dtype=torch.long
                )
            }

    def compute_metrics(prediction: Any) -> dict[str, float]:
        logits, labels = prediction
        predicted_ids = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            predicted_ids,
            labels=range(len(LABELS)),
            zero_division=0,
        )
        metrics = {
            "macro_f1": float(
                f1_score(labels, predicted_ids, labels=range(len(LABELS)), average="macro", zero_division=0)
            )
        }
        for index, label in enumerate(LABELS):
            metrics[f"{label.lower()}_precision"] = float(precision[index])
            metrics[f"{label.lower()}_recall"] = float(recall[index])
            metrics[f"{label.lower()}_f1"] = float(f1[index])
        return metrics

    training_args = TrainingArguments(
        output_dir=str(output / "trainer"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=25,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        fp16=True,
        gradient_checkpointing=True,
        seed=args.seed,
        data_seed=args.seed,
        report_to=[],
        dataloader_num_workers=0,
        optim="adamw_torch",
    )
    trainer_class = Trainer
    if class_weights is not None:
        class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

        class ClassWeightedTrainer(Trainer):
            def compute_loss(
                self,
                model: Any,
                inputs: dict[str, torch.Tensor],
                return_outputs: bool = False,
                num_items_in_batch: torch.Tensor | None = None,
            ) -> Any:
                labels = inputs["labels"]
                model_inputs = {key: value for key, value in inputs.items() if key != "labels"}
                outputs = model(**model_inputs)
                loss = torch.nn.functional.cross_entropy(
                    outputs.logits.reshape(-1, len(LABELS)),
                    labels.reshape(-1),
                    weight=class_weight_tensor.to(outputs.logits.device),
                )
                return (loss, outputs) if return_outputs else loss

        trainer_class = ClassWeightedTrainer

    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=VulnerabilityDataset(train_examples),
        eval_dataset=VulnerabilityDataset(validation_examples),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )
    trainer.loss_is_scaled_for_ga = loss_is_scaled_for_gradient_accumulation(args.class_weighting)
    trainer.train()
    model_output = output / "checkpoint"
    trainer.save_model(model_output)
    tokenizer.save_pretrained(model_output)
    (model_output / "label_mapping.json").write_text(json.dumps(label_to_id, indent=2), encoding="utf-8")
    (output / "training_log.json").write_text(
        json.dumps(trainer.state.log_history, indent=2), encoding="utf-8"
    )
    final_validation = trainer.evaluate()
    manifest.update(
        {
            "status": "trained",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_validation_metric": trainer.state.best_metric,
            "final_validation_metrics": final_validation,
            "training_runtime_seconds": next(
                (
                    entry["train_runtime"]
                    for entry in reversed(trainer.state.log_history)
                    if "train_runtime" in entry
                ),
                None,
            ),
            "checkpoint_path": str(model_output),
            "checkpoint_files": {
                path.name: sha256_file(path) for path in model_output.iterdir() if path.is_file()
            },
            "log_history": trainer.state.log_history,
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in (
        "status", "split_counts", "best_checkpoint", "best_validation_metric",
        "training_runtime_seconds", "checkpoint_path",
    )}, indent=2))


if __name__ == "__main__":
    main()
