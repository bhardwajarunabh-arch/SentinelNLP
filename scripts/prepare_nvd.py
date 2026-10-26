import argparse
from pathlib import Path

from backend.app.data_pipeline import create_splits, fetch_nvd, load_records


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and split eligible NVD CVEs.")
    parser.add_argument("--target-count", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records_path = Path("data/processed/nvd_records.json")
    fetch_nvd(records_path, target_count=args.target_count)
    create_splits(load_records(records_path), Path("data/splits/cve_splits.json"), seed=args.seed)


if __name__ == "__main__":
    main()
