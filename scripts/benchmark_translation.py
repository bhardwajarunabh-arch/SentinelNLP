"""Benchmark local NLLB translation without creating evaluation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from backend.app.data_pipeline import load_records
from backend.app.translations import (
    TRANSLATION_LANGUAGES,
    translation_identifier_report,
    validate_translation_for_language,
)
from scripts.translate_expanded_test import LANGUAGE_CODES, _translate_batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark local NLLB generation.")
    parser.add_argument("--records", type=Path, default=Path("data/expanded/nvd_records.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/expanded/cve_splits.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/translation_benchmark.json"))
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--cve-ids", nargs="+", required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    records = {record.cve_id: record for record in load_records(args.records)}
    test_ids = set(json.loads(args.splits.read_text(encoding="utf-8"))["splits"]["test"])
    invalid = [cve_id for cve_id in args.cve_ids if cve_id not in test_ids or cve_id not in records]
    if invalid:
        raise ValueError(f"CVE IDs must belong to the current test split: {invalid}")

    torch.set_num_threads(args.threads)
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.eval()
    load_seconds = time.perf_counter() - load_started

    rows: list[dict[str, object]] = []
    for language in TRANSLATION_LANGUAGES:
        for cve_id in args.cve_ids:
            text = records[cve_id].description
            token_started = time.perf_counter()
            token_count = len(tokenizer(text, truncation=True, max_length=512)["input_ids"])
            token_seconds = time.perf_counter() - token_started
            generation_started = time.perf_counter()
            translated = _translate_batch(
                tokenizer,
                model,
                [text],
                LANGUAGE_CODES[language],
                args.batch_size,
                args.max_length,
                args.num_beams,
            )[0]
            generation_seconds = time.perf_counter() - generation_started
            validation_status, _ = validate_translation_for_language(
                text, translated, language
            )
            identifier_report = translation_identifier_report(text, translated)
            rows.append(
                {
                    "cve_id": cve_id,
                    "language": language,
                    "severity": records[cve_id].severity,
                    "source_words": len(text.split()),
                    "source_tokens": token_count,
                    "tokenization_seconds": token_seconds,
                    "generation_seconds": generation_seconds,
                    "translated_characters": len(translated),
                    "validation_status": validation_status,
                    "preserved_identifiers": identifier_report["preserved"],
                    "missing_identifiers": identifier_report["missing"],
                    "translation_sample": translated[:400],
                }
            )

    total_generation = sum(float(row["generation_seconds"]) for row in rows)
    rejected_count = sum(row["validation_status"] != "valid" for row in rows)
    pair_seconds = total_generation / len(rows) if rows else None
    result = {
        "status": "benchmark_complete",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_revision": getattr(model.config, "_commit_hash", None) or "local_cached_revision_unknown",
        "settings": {
            "threads": args.threads,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "num_beams": args.num_beams,
            "do_sample": False,
            "inference_mode": True,
        },
        "cve_ids": args.cve_ids,
        "languages": list(TRANSLATION_LANGUAGES),
        "model_load_seconds": load_seconds,
        "total_tokenization_seconds": sum(float(row["tokenization_seconds"]) for row in rows),
        "total_generation_seconds": total_generation,
        "pair_count": len(rows),
        "seconds_per_pair": total_generation / len(rows) if rows else None,
        "estimated_full_1625_pair_hours": pair_seconds * 1625 / 3600 if pair_seconds else None,
        "estimated_full_run": {
            "pair_count": 1625,
            "generation_hours": pair_seconds * 1625 / 3600 if pair_seconds else None,
            "with_one_retry_per_failed_pair_hours": (
                pair_seconds * 1625 * (1 + rejected_count / len(rows)) / 3600
                if pair_seconds else None
            ),
            "rejected_pairs_in_benchmark": rejected_count,
            "note": "Retry estimate applies the observed rejected-pair fraction once; validation and process startup are additional.",
        },
        "rows": rows,
        "validation_summary": {
            language: {
                "valid": sum(
                    row["language"] == language and row["validation_status"] == "valid"
                    for row in rows
                ),
                "rejected": sum(
                    row["language"] == language and row["validation_status"] != "valid"
                    for row in rows
                ),
            }
            for language in TRANSLATION_LANGUAGES
        },
        "limitations": [
            "This benchmark measures local CPU inference only.",
            "It does not establish translation quality or human review.",
            "The estimate excludes retries, validation, and process startup.",
        ],
        "provenance": {
            "records_sha256": hashlib.sha256(args.records.read_bytes()).hexdigest(),
            "splits_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
