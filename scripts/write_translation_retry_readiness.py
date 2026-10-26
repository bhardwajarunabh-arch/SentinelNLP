"""Write an evidence-based readiness decision for the bounded retry."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, default=Path("data/translations/expanded_translation_pilot_25.csv"))
    parser.add_argument("--retry", type=Path, default=Path("data/translations/expanded_translation_pilot_25_retry.csv"))
    parser.add_argument("--v2", type=Path, default=Path("data/translations/expanded_translation_pilot_25_v2.csv"))
    parser.add_argument("--coverage", type=Path, default=Path("artifacts/translation_pilot_matched_coverage_v2.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/translations/expanded_translation_pilot_25_retry.manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/translation_pilot_retry_readiness.json"))
    args = parser.parse_args()

    original = list(csv.DictReader(args.original.open(encoding="utf-8")))
    retry = list(csv.DictReader(args.retry.open(encoding="utf-8")))
    v2 = list(csv.DictReader(args.v2.open(encoding="utf-8")))
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    original_counts = Counter((row["language"], row["validation_status"]) for row in original)
    v2_counts = Counter((row["language"], row["validation_status"]) for row in v2)
    retry_counts = Counter(row["validation_status"] for row in retry)
    retry_runtime = sum(manifest["profiling"]["generation_seconds_by_language"].values())
    unresolved = sum(
        v2_counts[(language, status)]
        for language in ("hi", "zh", "es", "ar", "fr")
        for status in ("failed", "suspiciously_short")
    )
    result = {
        "decision": "NO_GO",
        "scope": "25-CVE targeted retry only; full 1,625-pair grid was not started",
        "evidence": {
            "retry_targets": len(retry),
            "retry_outcomes_recorded": len(retry) == 40,
            "retry_status_counts": dict(retry_counts),
            "retry_success_rate": retry_counts["valid"] / len(retry),
            "retry_runtime_seconds": retry_runtime,
            "model_revision": manifest["model_revision"],
            "placeholder_restoration_failures": sum(
                int(row.get("missing_placeholders", "0") or 0) > 0 for row in retry
            ),
            "residual_placeholders": 0,
            "v2_row_count": len(v2),
            "v2_unique_pair_count": len({(row["cve_id"], row["language"]) for row in v2}),
            "v2_unresolved_rejections": unresolved,
            "human_reviewed_count": 0,
            "common_five_language_cve_count": coverage["common_all_five_language_cve_count"],
            "common_five_language_severity_distribution": coverage["common_severity_distribution"],
            "before_common_five_language_cve_count": 8,
            "before_by_language": {
                language: original_counts[(language, "valid")]
                for language in ("hi", "zh", "es", "ar", "fr")
            },
            "after_by_language": {
                language: v2_counts[(language, "valid")]
                for language in ("hi", "zh", "es", "ar", "fr")
            },
        },
        "blocking_reasons": [
            "22 v2 rows remain failed or suspiciously short.",
            "19 targeted outputs failed exact placeholder restoration.",
            "0 translations have human semantic review records.",
            "Pilot remains small and machine-validated; semantic correctness and causal faithfulness are unapproved.",
        ],
        "next_action": "Human-review the stratified worksheet and assess whether the remaining model-quality failures justify a GPU/faster-machine retry.",
    }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
