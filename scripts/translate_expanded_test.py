"""Generate local NLLB translations for the expanded held-out CVE split.

The command is resumable and never writes the existing translation template.
Generated translations are machine-translated and require automated validation
and human review before being treated as research-quality translations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
import re
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from huggingface_hub.constants import HF_HUB_CACHE
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from backend.app.data_pipeline import load_records
from backend.app.translations import (
    TRANSLATION_LANGUAGES,
    protect_technical_identifiers,
    restore_technical_identifiers,
    translation_identifier_report,
    validate_translation_for_language,
)

LANGUAGE_CODES = {
    "hi": "hin_Deva",
    "zh": "zho_Hans",
    "es": "spa_Latn",
    "ar": "arb_Arab",
    "fr": "fra_Latn",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolved_revision(model_name: str, requested_revision: str) -> str:
    if len(requested_revision) == 40 and all(
        character in "0123456789abcdef" for character in requested_revision.lower()
    ):
        return requested_revision
    cache_name = f"models--{model_name.replace('/', '--')}"
    ref_path = Path(HF_HUB_CACHE) / cache_name / "refs" / requested_revision
    if ref_path.is_file():
        return ref_path.read_text(encoding="utf-8").strip()
    return requested_revision


def _read_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {(row["cve_id"], row["language"]): row for row in csv.DictReader(handle)}


def _read_pair_manifest(path: Path | None) -> set[tuple[str, str]] | None:
    if path is None:
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"cve_id", "language"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("Pair manifest must contain cve_id and language columns")
        pairs = {(row["cve_id"], row["language"].lower()) for row in reader}
    if len(pairs) != sum(1 for _ in csv.DictReader(path.open(newline="", encoding="utf-8"))):
        raise ValueError("Pair manifest contains duplicate CVE-language pairs")
    return pairs


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "cve_id", "language", "severity", "model", "model_revision",
        "source_sha256",
        "original_text", "translated_text", "provenance", "validation_status",
        "validation_identifiers", "missing_identifiers", "source_token_count",
        "source_truncated", "output_token_count", "output_truncated",
        "chunk_count", "placeholder_count", "missing_placeholders",
        "generation_attempts", "generation_seconds",
        "generated_at", "error", "placeholder_error",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pending_cve_ids(
    existing: dict[tuple[str, str], dict[str, str]],
    test_ids: list[str],
    language: str,
    retry_failed: bool = False,
) -> list[str]:
    return [
        cve_id
        for cve_id in test_ids
        if (cve_id, language) not in existing
        or (
            retry_failed
            and existing[(cve_id, language)].get("validation_status") != "valid"
        )
    ]


def _translate_batch(
    tokenizer,
    model,
    texts: list[str],
    target_code: str,
    batch_size: int,
    max_length: int,
    num_beams: int,
    source_max_length: int = 512,
) -> list[str]:
    tokenizer.src_lang = "eng_Latn"
    results: list[str] = []
    target_id = tokenizer.convert_tokens_to_ids(target_code)
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(
            texts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=source_max_length,
        )
        with torch.inference_mode():
            generated = model.generate(
                **batch,
                forced_bos_token_id=target_id,
                max_length=max_length,
                num_beams=num_beams,
                do_sample=False,
            )
        results.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return results


def _chunk_text(text: str, tokenizer, max_source_tokens: int) -> list[str]:
    """Split long text deterministically, preferring sentence boundaries."""
    if len(tokenizer(text, add_special_tokens=True)["input_ids"]) <= max_source_tokens:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        candidate = " ".join(current + [sentence]).strip()
        if current and len(tokenizer(candidate, add_special_tokens=True)["input_ids"]) > max_source_tokens:
            chunks.append(" ".join(current))
            current = [sentence]
        else:
            current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    result: list[str] = []
    for chunk in chunks:
        ids = tokenizer(chunk, add_special_tokens=True)["input_ids"]
        if len(ids) <= max_source_tokens:
            result.append(chunk)
            continue
        words = chunk.split()
        for start in range(0, len(words), max(1, max_source_tokens // 2)):
            candidate_words = words[start : start + max(1, max_source_tokens // 2)]
            result.append(" ".join(candidate_words))
    return result


def _translate_with_chunking(
    tokenizer,
    model,
    text: str,
    target_code: str,
    max_source_tokens: int,
    max_length: int,
    num_beams: int,
) -> tuple[str, dict[str, object]]:
    protected_text, replacements = protect_technical_identifiers(text)
    chunks = _chunk_text(protected_text, tokenizer, max_source_tokens)
    translated_chunks = _translate_batch(
        tokenizer, model, chunks, target_code, len(chunks), max_length, num_beams,
        source_max_length=max_source_tokens,
    )
    translated_protected = " ".join(
        part.strip() for part in translated_chunks if part.strip()
    )
    translated, missing_placeholders = restore_technical_identifiers(
        translated_protected, replacements
    )
    output_tokens = len(tokenizer(translated, add_special_tokens=False)["input_ids"])
    return translated, {
        "source_token_count": len(tokenizer(text, add_special_tokens=True)["input_ids"]),
        "source_truncated": False,
        "chunk_count": len(chunks),
        "output_token_count": output_tokens,
        "output_truncated": output_tokens >= max_length,
        "placeholder_count": len(replacements),
        "missing_placeholders": len(missing_placeholders),
        "placeholder_error": ", ".join(missing_placeholders),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local NLLB translations for the expanded test split.")
    parser.add_argument("--records", type=Path, default=Path("data/expanded/nvd_records.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/expanded/cve_splits.json"))
    parser.add_argument("--output", type=Path, default=Path("data/translations/expanded_test_translations_generated.csv"))
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--limit-per-language", type=int, default=None)
    parser.add_argument("--cve-ids-file", type=Path, default=None)
    parser.add_argument("--pairs-file", type=Path, default=None,
                        help="CSV containing the exact cve_id/language pairs to process")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--source-max-length", type=int, default=512)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.limit_per_language is not None and args.limit_per_language < 1:
        raise ValueError("--limit-per-language must be positive")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.max_retries < 0:
        raise ValueError("--max-retries must not be negative")

    torch.set_num_threads(args.threads)
    records = {record.cve_id: record for record in load_records(args.records)}
    split_manifest = json.loads(args.splits.read_text(encoding="utf-8"))
    test_ids = list(split_manifest["splits"]["test"])
    if args.cve_ids_file is not None:
        selected = json.loads(args.cve_ids_file.read_text(encoding="utf-8"))
        test_ids = list(selected["cve_ids"])
    if args.limit_per_language is not None:
        test_ids = test_ids[: args.limit_per_language]
    requested_pairs = _read_pair_manifest(args.pairs_file)
    if requested_pairs is not None:
        invalid = [
            pair for pair in requested_pairs
            if pair[0] not in test_ids or pair[1] not in TRANSLATION_LANGUAGES
        ]
        if invalid:
            raise ValueError(f"Pair manifest contains invalid test-split/language pairs: {invalid[:3]}")
        test_ids = sorted({cve_id for cve_id, _ in requested_pairs})
    existing = _read_rows(args.output)
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model, revision=args.revision)
    model.eval()
    model_load_seconds = time.perf_counter() - load_started
    model_name = args.model
    model_revision = (
        getattr(model.config, "_commit_hash", None)
        or _resolved_revision(args.model, args.revision)
    )
    existing = {
        key: row for key, row in existing.items()
        if row.get("model_revision") in {None, "", model_revision}
        and (
            not row.get("source_sha256")
            or row["source_sha256"] == hashlib.sha256(
                records[key[0]].description.encode("utf-8")
            ).hexdigest()
        )
    }

    for language in TRANSLATION_LANGUAGES:
        language_ids = [
            cve_id for cve_id in test_ids
            if requested_pairs is None or (cve_id, language) in requested_pairs
        ]
        pending = pending_cve_ids(existing, language_ids, language, args.retry_failed)
        translations: dict[str, str] = {}
        generation_metadata: dict[str, dict[str, str]] = {}
        generation_started = time.perf_counter()
        for start in range(0, len(pending), args.batch_size):
            batch_ids = pending[start : start + args.batch_size]
            attempts = 0
            while attempts <= args.max_retries:
                attempts += 1
                try:
                    protected_groups: list[tuple[list[str], dict[str, str]]] = []
                    chunk_groups = []
                    for cve_id in batch_ids:
                        protected_text, replacements = protect_technical_identifiers(
                            records[cve_id].description
                        )
                        protected_chunks = _chunk_text(
                            protected_text, tokenizer, args.source_max_length
                        )
                        protected_groups.append((protected_chunks, replacements))
                        chunk_groups.append(protected_chunks)
                    flat_chunks = [chunk for group in chunk_groups for chunk in group]
                    flat_translations = _translate_batch(
                        tokenizer, model, flat_chunks, LANGUAGE_CODES[language],
                        args.batch_size, args.max_length, args.num_beams,
                        source_max_length=args.source_max_length,
                    )
                    offset = 0
                    for cve_id, chunks, (_, replacements) in zip(
                        batch_ids, chunk_groups, protected_groups, strict=True
                    ):
                        translated_chunks = flat_translations[offset : offset + len(chunks)]
                        offset += len(chunks)
                        translated_protected = " ".join(
                            part.strip() for part in translated_chunks if part.strip()
                        )
                        text, missing_placeholders = restore_technical_identifiers(
                            translated_protected, replacements
                        )
                        source_count = len(tokenizer(records[cve_id].description, add_special_tokens=True)["input_ids"])
                        output_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                        translations[cve_id] = text
                        generation_metadata[cve_id] = {
                            "source_token_count": str(source_count),
                            "source_truncated": "false",
                            "output_token_count": str(output_count),
                            "output_truncated": str(output_count >= args.max_length * len(chunks)).lower(),
                            "chunk_count": str(len(chunks)),
                            "placeholder_count": str(len(replacements)),
                            "missing_placeholders": str(len(missing_placeholders)),
                            "placeholder_error": ", ".join(missing_placeholders),
                            "generation_attempts": str(attempts),
                            "error": "",
                        }
                    break
                except (RuntimeError, ValueError, IndexError) as exc:
                    if attempts > args.max_retries:
                        for cve_id in batch_ids:
                            translations[cve_id] = ""
                            generation_metadata[cve_id] = {
                                "source_token_count": "",
                                "source_truncated": "",
                                "output_token_count": "0",
                                "output_truncated": "",
                                "chunk_count": "0",
                                "placeholder_count": "0",
                                "missing_placeholders": "0",
                                "placeholder_error": "",
                                "generation_attempts": str(attempts),
                                "error": f"{type(exc).__name__}: {exc}",
                            }
        for cve_id in language_ids:
            record = records[cve_id]
            old = existing.get((cve_id, language))
            if old and cve_id not in translations and not (
                args.retry_failed and old.get("validation_status") != "valid"
            ):
                continue
            text = translations.get(cve_id, "")
            status, identifiers = validate_translation_for_language(record.description, text, language)
            existing[(cve_id, language)] = {
                "cve_id": cve_id,
                "language": language,
                "severity": record.severity,
                "model": model_name,
                "model_revision": model_revision,
                "source_sha256": hashlib.sha256(record.description.encode("utf-8")).hexdigest(),
                "original_text": record.description,
                "translated_text": text,
                "provenance": "local_nllb_generation",
                "validation_status": status,
                "validation_identifiers": "|".join(identifiers),
                "missing_identifiers": "|".join(
                    translation_identifier_report(record.description, text)["missing"]
                ),
                "source_token_count": generation_metadata.get(cve_id, {}).get("source_token_count", ""),
                "source_truncated": generation_metadata.get(cve_id, {}).get("source_truncated", ""),
                "output_token_count": generation_metadata.get(cve_id, {}).get("output_token_count", "0"),
                "output_truncated": generation_metadata.get(cve_id, {}).get("output_truncated", ""),
                "chunk_count": generation_metadata.get(cve_id, {}).get("chunk_count", "0"),
                "placeholder_count": generation_metadata.get(cve_id, {}).get("placeholder_count", "0"),
                "missing_placeholders": generation_metadata.get(cve_id, {}).get("missing_placeholders", "0"),
                "generation_attempts": generation_metadata.get(cve_id, {}).get("generation_attempts", "0"),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "error": generation_metadata.get(cve_id, {}).get("error", ""),
                "placeholder_error": generation_metadata.get(cve_id, {}).get("placeholder_error", ""),
            }
            if generation_metadata.get(cve_id, {}).get("missing_placeholders", "0") != "0":
                existing[(cve_id, language)]["validation_status"] = "failed"
                existing[(cve_id, language)]["error"] = (
                    "Placeholder restoration failed: "
                    + generation_metadata[cve_id].get("placeholder_error", "")
                )
        language_seconds = time.perf_counter() - generation_started
        for cve_id in pending:
            existing[(cve_id, language)]["generation_seconds"] = f"{language_seconds / len(pending):.6f}"
        _write_rows(
            args.output,
            [
                existing[key]
                for key in sorted(existing)
                if (
                    key[0] in test_ids and key[1] in TRANSLATION_LANGUAGES
                    and (requested_pairs is None or key in requested_pairs)
                )
            ],
        )

    final_rows = [
        row for key, row in existing.items()
        if requested_pairs is None or key in requested_pairs
    ]
    manifest = {
        "status": "machine_translations_generated",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "model_revision": model_revision,
        "language_codes": LANGUAGE_CODES,
        "generation": {
            "num_beams": args.num_beams,
            "do_sample": False,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "source_max_length": args.source_max_length,
            "threads": args.threads,
            "max_retries": args.max_retries,
        },
        "profiling": {
            "model_load_seconds": model_load_seconds,
            "generation_seconds_by_language": {
                language: sum(
                    float(row.get("generation_seconds", 0))
                    for row in final_rows
                    if row.get("language") == language
                )
                for language in TRANSLATION_LANGUAGES
            },
            "note": "Generation timing includes tokenization and model.generate for each language batch.",
        },
        "records_sha256": _sha256(args.records),
        "splits_sha256": _sha256(args.splits),
        "test_cve_ids": test_ids,
        "row_count": len(final_rows),
        "counts_by_status": {
            status: sum(row.get("validation_status") == status for row in final_rows)
            for status in ("valid", "needs_review", "failed", "missing", "technical_identifier_missing")
        },
        "dependency_versions": {
            "torch": version("torch"),
            "transformers": version("transformers"),
            "sentencepiece": version("sentencepiece"),
        },
        "human_review": False,
        "note": "Translations are machine-generated and automated validation is not human translation-quality review.",
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
