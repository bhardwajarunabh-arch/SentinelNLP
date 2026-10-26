import argparse
import hashlib
import json
from pathlib import Path
from importlib.metadata import version
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.baseline import train_baseline
from backend.app.data_pipeline import load_records


def training_records(records, split_manifest):
    records_by_id = {record.cve_id: record for record in records}
    train_ids = split_manifest["splits"]["train"]
    if set(train_ids) - set(records_by_id):
        raise ValueError("Split manifest contains CVE IDs missing from the dataset")
    return [records_by_id[cve_id] for cve_id in train_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the English TF-IDF baseline.")
    parser.add_argument("--data", type=Path, default=Path("data/processed/nvd_records.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/splits/cve_splits.json"))
    parser.add_argument("--output", type=Path, default=Path("models/baseline"))
    args = parser.parse_args()
    records = load_records(args.data)
    split_manifest = json.loads(args.splits.read_text(encoding="utf-8"))
    train_ids = split_manifest["splits"]["train"]
    train_records = training_records(records, split_manifest)
    metadata = train_baseline([record.description for record in train_records],
                              [record.severity for record in train_records], args.output)
    model_path = args.output / "model.joblib"
    metadata.update(
        {
            "training_cve_ids": train_ids,
            "training_record_count": len(train_records),
            "dataset_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
            "split_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
            "label_order": metadata["classes"],
            "random_seed": 42,
            "dependency_versions": {
                "python": __import__("sys").version.split()[0],
                "scikit-learn": version("scikit-learn"),
                "joblib": version("joblib"),
            },
        }
    )
    metadata["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    (args.output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
