"""Select a reproducible, stratified pilot from the fixed expanded test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.data_pipeline import load_records


def select(records_path: Path, splits_path: Path, per_severity: int = 6) -> list[str]:
    records = {record.cve_id: record for record in load_records(records_path)}
    test_ids = json.loads(splits_path.read_text(encoding="utf-8"))["splits"]["test"]
    selected: list[str] = []
    for severity in ("Critical", "High", "Medium", "Low"):
        candidates = [
            cve_id for cve_id in test_ids
            if records[cve_id].severity == severity
        ]
        candidates.sort(key=lambda cve_id: (len(records[cve_id].description), cve_id))
        if len(candidates) <= per_severity:
            selected.extend(candidates)
            continue
        indices = {
            round(index * (len(candidates) - 1) / (per_severity - 1))
            for index in range(per_severity)
        }
        selected.extend(candidates[index] for index in sorted(indices))
    if len(selected) < 25:
        remaining = [cve_id for cve_id in test_ids if cve_id not in selected]
        remaining.sort(key=lambda cve_id: (-len(records[cve_id].description), cve_id))
        selected.append(remaining[0])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, default=Path("data/expanded/nvd_records.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/expanded/cve_splits.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/translation_pilot_selection.json"))
    parser.add_argument("--per-severity", type=int, default=6)
    args = parser.parse_args()
    ids = select(args.records, args.splits, args.per_severity)
    payload = {
        "status": "pilot_selected",
        "strategy": "six evenly length-spaced CVEs within each severity class in fixed test order",
        "per_severity": args.per_severity,
        "cve_ids": ids,
        "pair_count": len(ids) * 5,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
