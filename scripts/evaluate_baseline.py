import argparse
import csv
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.baseline import predict
from backend.app.data_pipeline import load_records
from backend.app.metrics import classification_metrics
from backend.app.risk import critical_risk_metrics
LABELS = ["Critical", "High", "Low", "Medium"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the baseline on the untouched test split.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/nvd_records.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits/cve_splits.json"))
    parser.add_argument("--model", type=Path, default=Path("models/baseline/model.joblib"))
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/baseline_test_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/baseline_metrics.json"))
    args = parser.parse_args()

    records = {record.cve_id: record for record in load_records(args.data)}
    manifest = json.loads(args.splits.read_text(encoding="utf-8"))
    test_ids = manifest["splits"]["test"]
    if len(test_ids) != len(set(test_ids)) or set(test_ids) - set(records):
        raise ValueError("Test split contains duplicate or unknown CVE IDs")
    test_records = [records[cve_id] for cve_id in test_ids]
    predictions, probabilities = predict(args.model, [record.description for record in test_records])
    model = __import__("joblib").load(args.model)
    rows = []
    for record, prediction, probability in zip(test_records, predictions, probabilities):
        rows.append(
            {
                "cve_id": record.cve_id,
                "true_label": record.severity,
                "predicted_label": prediction,
                **{f"probability_{label}": float(probability[index]) for index, label in enumerate(model.classes_)},
            }
        )
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    with args.predictions.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    labels = [record.severity for record in test_records]
    result = classification_metrics(labels, predictions, probabilities, labels=LABELS)
    result["critical_risk"] = critical_risk_metrics(labels, predictions)
    result["status"] = "valid_train_only_test_evaluation"
    result["provenance"] = {
        "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
        "split_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "prediction_sha256": hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
        "test_cve_ids": test_ids,
        "label_order": LABELS,
        "dependency_versions": {"scikit-learn": version("scikit-learn"), "joblib": version("joblib")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
