"""Write a conservative RQ3 cross-language association report."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize cross-language RQ3 evidence.")
    parser.add_argument("--translations", type=Path, default=Path("artifacts/metrics.json"))
    parser.add_argument("--faithfulness", type=Path, default=Path("artifacts/faithfulness.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/rq3_analysis.json"))
    args = parser.parse_args()
    translation = json.loads(args.translations.read_text(encoding="utf-8"))
    faithfulness = json.loads(args.faithfulness.read_text(encoding="utf-8"))
    result = {
        "status": "blocked_insufficient_cross_language_data",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "denominators": {
            "translation_matched_cves": translation.get("sample_count", 0),
            "languages_with_results": [
                language
                for language, value in translation.get("languages", {}).items()
                if value.get("sample_count", 0)
            ],
            "english_faithfulness_cves": faithfulness.get("sample_count", 0),
        },
        "contingency_tables": {},
        "association_tests": {},
        "effect_sizes": {},
        "limitation": "Cross-language disagreement, translation faithfulness, and Critical underprediction cannot be joined without reliable matched translations.",
        "interpretation": "No association or causal claim is supported; multilingual statistical power is zero.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
