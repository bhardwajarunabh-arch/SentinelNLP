"""Run matched baseline, RQ1 metrics, and RQ2 diagnostics on eligible pilot rows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

from backend.app.data_pipeline import load_records
from backend.app.faithfulness import area_under_curve, perturbation_curve
from backend.app.translations import validate_translation_for_language

LABELS = ["Critical", "High", "Low", "Medium"]
LANGUAGES = ["hi", "zh", "es", "ar", "fr"]


def tokens_for(text: str, language: str) -> list[str]:
    if language == "zh":
        return re.findall(r"[\u3400-\u9fff]|[A-Za-z0-9_.:/-]+|[^\w\s]", text)
    return re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)


def ci(successes: int, total: int) -> list[float] | None:
    if not total:
        return None
    p = successes / total
    margin = 1.96 * (p * (1 - p) / total) ** 0.5
    return [max(0.0, p - margin), min(1.0, p + margin)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--translations", type=Path, required=True)
    parser.add_argument("--records", type=Path, default=Path("data/expanded/nvd_records.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/expanded/cve_splits.json"))
    parser.add_argument("--model", type=Path, default=Path("models/expanded_baseline/model.joblib"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--num-features", type=int, default=10)
    parser.add_argument("--num-samples", type=int, default=300)
    args = parser.parse_args()

    records = {record.cve_id: record for record in load_records(args.records)}
    test_ids = set(json.loads(args.splits.read_text(encoding="utf-8"))["splits"]["test"])
    rows = list(csv.DictReader(args.translations.open(newline="", encoding="utf-8")))
    eligible = []
    exclusions = Counter()
    seen = set()
    for row in rows:
        pair = (row.get("cve_id", ""), row.get("language", "").lower())
        if pair in seen:
            exclusions["duplicate"] += 1
            continue
        seen.add(pair)
        if row.get("validation_status") != "valid":
            exclusions[row.get("validation_status", "missing")] += 1
            continue
        if pair[0] not in test_ids or pair[0] not in records:
            exclusions["outside_test_split"] += 1
            continue
        if row.get("original_text") != records[pair[0]].description:
            exclusions["source_mismatch"] += 1
            continue
        computed, _ = validate_translation_for_language(
            records[pair[0]].description, row.get("translated_text", ""), pair[1]
        )
        if computed != "valid":
            exclusions[computed] += 1
            continue
        eligible.append(row)

    model = joblib.load(args.model)
    explainer = None
    try:
        from lime.lime_text import LimeTextExplainer
        explainer = LimeTextExplainer(class_names=list(model.classes_), random_state=42)
    except ImportError:
        exclusions["lime_not_installed"] += len(eligible)

    english_by_id = {}
    for cve_id in sorted({row["cve_id"] for row in eligible}):
        record = records[cve_id]
        english_by_id[cve_id] = {
            "prediction": model.predict([record.description])[0],
            "probabilities": model.predict_proba([record.description])[0].tolist(),
        }
    prediction_rows = []
    faithfulness_rows = []
    for row in eligible:
        cve_id = row["cve_id"]
        language = row["language"].lower()
        text = row["translated_text"]
        prediction = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0].tolist()
        output = {
            "cve_id": cve_id,
            "language": language,
            "true_label": records[cve_id].severity,
            "predicted_label": prediction,
            "english_predicted_label": english_by_id[cve_id]["prediction"],
            "agreement": prediction == english_by_id[cve_id]["prediction"],
            "validation_status": row["validation_status"],
            "review_status": "unreviewed",
            "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "probability_Critical": probabilities[list(model.classes_).index("Critical")],
            "probability_High": probabilities[list(model.classes_).index("High")],
            "probability_Low": probabilities[list(model.classes_).index("Low")],
            "probability_Medium": probabilities[list(model.classes_).index("Medium")],
        }
        prediction_rows.append(output)
        if explainer is None:
            continue
        try:
            label_index = list(model.classes_).index(prediction)
            explanation = explainer.explain_instance(
                text, model.predict_proba, labels=[label_index],
                num_features=args.num_features, num_samples=args.num_samples,
            )
            segments = explanation.as_list(label=label_index)
            tokens = tokens_for(text, language)
            ranked = []
            used = set()
            features = []
            for rank, (feature, weight) in enumerate(segments):
                features.append({"feature": feature, "weight": float(weight), "rank": rank})
                for index, token in enumerate(tokens):
                    if index not in used and token == feature:
                        ranked.append(index)
                        used.add(index)
                        break
            probability = lambda value: float(model.predict_proba([value])[0][label_index])
            deletion = perturbation_curve(tokens, ranked, probability, "deletion")
            insertion = perturbation_curve(tokens, ranked, probability, "insertion")
            faithfulness_rows.append({
                "cve_id": cve_id,
                "language": language,
                "true_label": records[cve_id].severity,
                "predicted_label": prediction,
                "target_probability": probabilities[label_index],
                "features": features,
                "deletion_auc": area_under_curve(deletion),
                "insertion_auc": area_under_curve(insertion),
                "deletion_probability_drop": deletion[0]["probability"] - deletion[-1]["probability"],
                "insertion_probability_gain": insertion[-1]["probability"] - insertion[0]["probability"],
                "deletion_baseline_probability": deletion[-1]["probability"],
                "insertion_baseline_probability": insertion[0]["probability"],
                "matched_feature_count": len(ranked),
                "error": "",
            })
        except (ValueError, RuntimeError, IndexError, FloatingPointError) as exc:
            faithfulness_rows.append({
                "cve_id": cve_id, "language": language, "true_label": records[cve_id].severity,
                "predicted_label": prediction, "error": f"{type(exc).__name__}: {exc}",
            })

    def metrics(language_rows):
        truth = [r["true_label"] for r in language_rows]
        pred = [r["predicted_label"] for r in language_rows]
        return {
            "sample_count": len(language_rows),
            "coverage_denominator": len({r["cve_id"] for r in rows if r["language"] == language_rows[0]["language"]}) if language_rows else 0,
            "accuracy": float(accuracy_score(truth, pred)) if truth else None,
            "accuracy_95_ci": ci(sum(a == b for a, b in zip(truth, pred)), len(truth)),
            "macro_f1": float(f1_score(truth, pred, labels=LABELS, average="macro", zero_division=0)) if truth else None,
            "weighted_f1": float(f1_score(truth, pred, labels=LABELS, average="weighted", zero_division=0)) if truth else None,
            "balanced_accuracy": float(balanced_accuracy_score(truth, pred)) if truth else None,
            "labels": LABELS,
            "confusion_matrix": confusion_matrix(truth, pred, labels=LABELS).tolist() if truth else [],
            "per_class": classification_report(truth, pred, labels=LABELS, output_dict=True, zero_division=0) if truth else {},
            "agreement_count": sum(r["agreement"] for r in language_rows),
            "agreement_rate": sum(r["agreement"] for r in language_rows) / len(language_rows) if language_rows else None,
            "disagreement_count": sum(not r["agreement"] for r in language_rows),
            "severity_transition_matrix": {
                actual: {
                    predicted: sum(
                        r["true_label"] == actual and r["predicted_label"] == predicted
                        for r in language_rows
                    )
                    for predicted in LABELS
                }
                for actual in LABELS
            },
            "critical_risk": {
                "actual_critical_count": sum(r["true_label"] == "Critical" for r in language_rows),
                "predicted_critical_count": sum(r["predicted_label"] == "Critical" for r in language_rows),
                "critical_true_positive": sum(
                    r["true_label"] == "Critical" and r["predicted_label"] == "Critical"
                    for r in language_rows
                ),
                "critical_underprediction_count": sum(
                    r["true_label"] == "Critical" and r["predicted_label"] != "Critical"
                    for r in language_rows
                ),
                "critical_to_high": sum(
                    r["true_label"] == "Critical" and r["predicted_label"] == "High"
                    for r in language_rows
                ),
                "critical_to_medium": sum(
                    r["true_label"] == "Critical" and r["predicted_label"] == "Medium"
                    for r in language_rows
                ),
                "critical_to_low": sum(
                    r["true_label"] == "Critical" and r["predicted_label"] == "Low"
                    for r in language_rows
                ),
            },
        }

    rq1 = {
        "status": "exploratory_pilot",
        "sample_count": len(eligible),
        "expected_pairs": len(json.loads(json.dumps(list(test_ids)))) * len(LANGUAGES),
        "excluded_rows": dict(exclusions),
        "languages": {
            language: metrics([r for r in prediction_rows if r["language"] == language])
            for language in LANGUAGES
        },
    }
    ids_by_language = {
        language: {row["cve_id"] for row in prediction_rows if row["language"] == language}
        for language in LANGUAGES
    }
    common_ids = set.intersection(*(ids_by_language[language] for language in LANGUAGES))
    common_rows = {
        language: {
            row["cve_id"]: row
            for row in prediction_rows
            if row["language"] == language and row["cve_id"] in common_ids
        }
        for language in LANGUAGES
    }
    rq1["paired_common_intersection"] = {
        "cve_count": len(common_ids),
        "cve_ids": sorted(common_ids),
        "languages": {
            language: {
                "sample_count": len(common_rows[language]),
                "accuracy": sum(
                    row["true_label"] == row["predicted_label"]
                    for row in common_rows[language].values()
                ) / len(common_rows[language]) if common_rows[language] else None,
                "agreement_with_english": sum(
                    row["predicted_label"] == row["english_predicted_label"]
                    for row in common_rows[language].values()
                ) / len(common_rows[language]) if common_rows[language] else None,
                "disagreement_with_english_count": sum(
                    row["predicted_label"] != row["english_predicted_label"]
                    for row in common_rows[language].values()
                ),
            }
            for language in LANGUAGES
        },
    }
    rq2 = {
        "status": "exploratory_pilot",
        "settings": {"method": "LIME", "num_features": args.num_features, "num_samples": args.num_samples,
                     "tokenization": "language-aware regex; Chinese character segmentation; empty-text baselines"},
        "sample_count": len(faithfulness_rows),
        "languages": {},
    }
    for language in LANGUAGES:
        selected = [r for r in faithfulness_rows if r["language"] == language and not r.get("error")]
        rq2["languages"][language] = {
            "sample_count": len(selected),
            "mean_deletion_auc": statistics.fmean(r["deletion_auc"] for r in selected) if selected else None,
            "mean_insertion_auc": statistics.fmean(r["insertion_auc"] for r in selected) if selected else None,
            "mean_deletion_probability_drop": statistics.fmean(r["deletion_probability_drop"] for r in selected) if selected else None,
            "mean_insertion_probability_gain": statistics.fmean(r["insertion_probability_gain"] for r in selected) if selected else None,
            "critical_count": sum(r["true_label"] == "Critical" for r in selected),
            "errors": sum(r.get("error", "") != "" for r in faithfulness_rows if r["language"] == language),
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in [
        ("multilingual_pilot_predictions.json", {
            "status": "exploratory_pilot",
            "label_order": LABELS,
            "records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
            "splits_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
            "translations_sha256": hashlib.sha256(args.translations.read_bytes()).hexdigest(),
            "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
            "training_test_overlap": 0,
            "rows": prediction_rows,
        }),
        ("multilingual_pilot_rq1.json", rq1),
        ("multilingual_pilot_rq2.json", rq2),
        ("multilingual_pilot_faithfulness_rows.json", {"status": "exploratory_pilot", "rows": faithfulness_rows}),
    ]:
        (args.output_dir / name).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"eligible": len(eligible), "excluded": dict(exclusions), "rq1": rq1["languages"], "rq2": rq2["languages"]}, indent=2))


if __name__ == "__main__":
    main()
