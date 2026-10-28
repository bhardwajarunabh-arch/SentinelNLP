"""Generate local NLLB translations for explicitly scoped fixed-split CVEs.

The command is resumable and never writes the existing translation template.
Generated translations are machine-translated and require automated validation
and human review before being treated as research-quality translations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import re
import subprocess
import unicodedata
from collections import Counter
from dataclasses import dataclass
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
    split_technical_placeholders,
    technical_identifier_occurrence_report,
    translation_identifier_report,
    validate_technical_placeholders,
    validate_translation_for_language,
)

LANGUAGE_CODES = {
    "hi": "hin_Deva",
    "zh": "zho_Hans",
    "es": "spa_Latn",
    "ar": "arb_Arab",
    "fr": "fra_Latn",
}


@dataclass(frozen=True)
class _TranslationPart:
    kind: str
    text: str
    leading_boundary: str = ""
    trailing_boundary: str = ""
    unit_index: int | None = None


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
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {(row["cve_id"], row["language"]): row for row in csv.DictReader(handle)}


def _read_pair_manifest(path: Path | None) -> set[tuple[str, str]] | None:
    if path is None:
        return None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"cve_id", "language"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError("Pair manifest must contain cve_id and language columns")
        rows = list(reader)
    pairs = {(row["cve_id"], row["language"].lower()) for row in rows}
    if len(pairs) != len(rows):
        raise ValueError("Pair manifest contains duplicate CVE-language pairs")
    return pairs


def ids_for_split_scope(splits: dict[str, list[str]], scope: str) -> list[str]:
    if scope == "train_validation":
        return list(splits["train"]) + list(splits["validation"])
    if scope in {"train", "validation", "test"}:
        return list(splits[scope])
    raise ValueError(f"Unsupported translation split scope: {scope}")


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    fields = [
        "cve_id", "language", "severity", "model", "model_revision",
        "source_sha256",
        "original_text", "translated_text", "provenance", "validation_status",
        "validation_identifiers", "missing_identifiers", "source_token_count",
        "source_truncated", "output_token_count", "output_truncated",
        "incomplete_units", "chunk_count", "placeholder_count", "missing_placeholders",
        "generation_attempts", "generation_seconds",
        "generated_at", "error", "validation_error", "placeholder_error",
        "identifier_integrity_error",
    ]
    with temporary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    for attempt in range(10):
        try:
            os.replace(temporary_path, path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.25 * (attempt + 1))


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
    device: torch.device | str = "cpu",
    return_completion: bool = False,
) -> list[str] | tuple[list[str], list[bool]]:
    tokenizer.src_lang = "eng_Latn"
    results = [""] * len(texts)
    completed = [False] * len(texts)
    target_id = tokenizer.convert_tokens_to_ids(target_code)
    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start : start + batch_size]
        batch = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=source_max_length,
        )
        input_lengths = batch["attention_mask"].sum(dim=1).tolist()
        caps: dict[int, list[int]] = {}
        for index, input_length in enumerate(input_lengths):
            per_unit_cap = _per_unit_generation_cap(int(input_length), max_length)
            caps.setdefault(per_unit_cap, []).append(index)
        eos_token_ids = tokenizer.eos_token_id
        eos_ids = (
            set(eos_token_ids)
            if isinstance(eos_token_ids, list)
            else {eos_token_ids}
        )
        for per_unit_cap, indexes in caps.items():
            for group_start in range(0, len(indexes), batch_size):
                group_indexes = indexes[group_start : group_start + batch_size]
                index_tensor = torch.tensor(group_indexes, device=batch["input_ids"].device)
                grouped_batch = {
                    key: value.index_select(0, index_tensor).to(device)
                    for key, value in batch.items()
                }
                with torch.inference_mode():
                    generated = model.generate(
                        **grouped_batch,
                        forced_bos_token_id=target_id,
                        max_new_tokens=per_unit_cap,
                        num_beams=num_beams,
                        do_sample=False,
                    )
                decoded = tokenizer.batch_decode(
                    generated, skip_special_tokens=True
                )
                for local_index, sequence in enumerate(generated):
                    batch_index = group_indexes[local_index]
                    result_index = start + batch_index
                    results[result_index] = decoded[local_index]
                    completed[result_index] = any(
                        int(token_id) in eos_ids for token_id in sequence[1:]
                    )
    if return_completion:
        return results, completed
    return results


def _per_unit_generation_cap(input_tokens: int, maximum_new_tokens: int) -> int:
    return min(maximum_new_tokens, max(32, input_tokens * 2))


def _is_boundary_character(character: str) -> bool:
    return character.isspace() or unicodedata.category(character).startswith("P")


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


def _prepare_translation_units(
    chunks: list[str], replacements: dict[str, str]
) -> tuple[list[str], list[list[_TranslationPart]]]:
    issues = validate_technical_placeholders(chunks, replacements)
    if issues:
        raise ValueError(f"Identifier placeholders were damaged by chunking: {issues}")

    units: list[str] = []
    layouts: list[list[_TranslationPart]] = []
    for chunk in chunks:
        layout: list[_TranslationPart] = []
        for kind, value in split_technical_placeholders(chunk):
            if kind == "placeholder":
                layout.append(_TranslationPart(kind="placeholder", text=value))
                continue

            if not value.strip():
                layout.append(_TranslationPart(kind="text", text=value))
                continue

            leading_length = 0
            while leading_length < len(value) and _is_boundary_character(
                value[leading_length]
            ):
                leading_length += 1
            trailing_start = len(value)
            while trailing_start > leading_length and _is_boundary_character(
                value[trailing_start - 1]
            ):
                trailing_start -= 1
            core = value[leading_length:trailing_start]
            if not core or not any(character.isalnum() for character in core):
                layout.append(_TranslationPart(kind="text", text=value))
                continue
            unit_index = len(units)
            units.append(core)
            layout.append(
                _TranslationPart(
                    kind="text",
                    text=core,
                    leading_boundary=value[:leading_length],
                    trailing_boundary=value[trailing_start:],
                    unit_index=unit_index,
                )
            )
        layouts.append(layout)
    return units, layouts


def _render_translated_chunks(
    layouts: list[list[_TranslationPart]], translated_units: list[str]
) -> str:
    translated_chunks: list[str] = []
    for layout in layouts:
        parts: list[str] = []
        for part in layout:
            if part.kind == "placeholder":
                parts.append(part.text)
                continue
            if part.unit_index is None:
                parts.append(part.text)
                continue
            translated = translated_units[part.unit_index].strip()
            if not translated:
                raise ValueError("NLLB returned an empty translation fragment")
            parts.append(
                part.leading_boundary + translated + part.trailing_boundary
            )
        translated_chunks.append("".join(parts).strip())
    return " ".join(chunk for chunk in translated_chunks if chunk)


def _translation_units_truncated(
    translated_units: list[str], completion_flags: list[bool]
) -> bool:
    if len(translated_units) != len(completion_flags):
        raise ValueError("Every generated text unit must have a completion result")
    return any(not completed for completed in completion_flags)


def _translate_with_chunking(
    tokenizer,
    model,
    text: str,
    target_code: str,
    max_source_tokens: int,
    max_length: int,
    num_beams: int,
    device: torch.device | str = "cpu",
    batch_size: int = 2,
) -> tuple[str, dict[str, object]]:
    protected_text, replacements = protect_technical_identifiers(text)
    chunks = _chunk_text(protected_text, tokenizer, max_source_tokens)
    units, layouts = _prepare_translation_units(chunks, replacements)
    translated_units, completion_flags = _translate_batch(
        tokenizer, model, units, target_code, batch_size, max_length,
        num_beams, source_max_length=max_source_tokens, device=device,
        return_completion=True,
    ) if units else ([], [])
    translated_protected = _render_translated_chunks(layouts, translated_units)
    translated, missing_placeholders = restore_technical_identifiers(
        translated_protected, replacements
    )
    identifier_integrity = technical_identifier_occurrence_report(text, translated)
    output_tokens = len(tokenizer(translated, add_special_tokens=False)["input_ids"])
    return translated, {
        "source_token_count": len(tokenizer(text, add_special_tokens=True)["input_ids"]),
        "source_truncated": False,
        "chunk_count": len(chunks),
        "output_token_count": output_tokens,
        "output_truncated": _translation_units_truncated(
            translated_units, completion_flags
        ),
        "incomplete_units": sum(not completed for completed in completion_flags),
        "placeholder_count": len(replacements),
        "missing_placeholders": len(missing_placeholders),
        "placeholder_error": ", ".join(missing_placeholders),
        "identifier_integrity_error": " | ".join(identifier_integrity["issues"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate local NLLB translations for a fixed split scope.")
    parser.add_argument("--records", type=Path, default=Path("data/expanded/nvd_records.json"))
    parser.add_argument("--splits", type=Path, default=Path("data/expanded/cve_splits.json"))
    parser.add_argument("--output", type=Path, default=Path("data/translations/expanded_test_translations_generated.csv"))
    parser.add_argument(
        "--split-scope",
        choices=("test", "train", "validation", "train_validation"),
        default="test",
        help="Fixed split membership required for selected CVE IDs.",
    )
    parser.add_argument("--model", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--limit-per-language", type=int, default=None)
    parser.add_argument("--cve-ids-file", type=Path, default=None)
    parser.add_argument("--pairs-file", type=Path, default=None,
                        help="CSV containing the exact cve_id/language pairs to process")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--source-max-length", type=int, default=512)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--gpu-report-output", type=Path, default=None)
    args = parser.parse_args()
    if args.limit_per_language is not None and args.limit_per_language < 1:
        raise ValueError("--limit-per-language must be positive")
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    if args.max_retries < 0:
        raise ValueError("--max-retries must not be negative")

    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing to fall back to CPU")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    tensor_check: dict[str, int] | None = None
    if device.type == "cuda":
        values = torch.arange(4096, device=device)
        tensor_check = {"elements": values.numel(), "squared_sum": int((values * values).sum().item())}
        torch.cuda.synchronize(device)

    records = {record.cve_id: record for record in load_records(args.records)}
    split_manifest = json.loads(args.splits.read_text(encoding="utf-8"))
    scope_ids = ids_for_split_scope(split_manifest["splits"], args.split_scope)
    selected_ids = list(scope_ids)
    if args.cve_ids_file is not None:
        selected = json.loads(args.cve_ids_file.read_text(encoding="utf-8"))
        selected_ids = list(selected["cve_ids"])
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("Selected CVE ID list contains duplicates")
    outside_scope = set(selected_ids) - set(scope_ids)
    if outside_scope:
        raise ValueError(
            f"Selected CVEs are outside the requested {args.split_scope} split scope: "
            f"{sorted(outside_scope)[:5]}"
        )
    if args.limit_per_language is not None:
        selected_ids = selected_ids[: args.limit_per_language]
    requested_pairs = _read_pair_manifest(args.pairs_file)
    if requested_pairs is not None:
        invalid = [
            pair for pair in requested_pairs
            if pair[0] not in selected_ids or pair[1] not in TRANSLATION_LANGUAGES
        ]
        if invalid:
            raise ValueError(
                f"Pair manifest contains CVEs outside the selected "
                f"{args.split_scope} split/language scope: {invalid[:3]}"
            )
        selected_ids = sorted({cve_id for cve_id, _ in requested_pairs})
    existing = _read_rows(args.output)
    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=args.revision,
        src_lang="eng_Latn",
        local_files_only=args.local_files_only,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model,
        revision=args.revision,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.eval()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
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

    generation_seconds_by_language: dict[str, float] = {}
    pairs_per_second_by_language: dict[str, float] = {}
    generated_pair_count = 0
    first_translation: dict[str, object] | None = None
    first_valid_translation: dict[str, object] | None = None
    run_started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for language in TRANSLATION_LANGUAGES:
        language_ids = [
            cve_id for cve_id in selected_ids
            if requested_pairs is None or (cve_id, language) in requested_pairs
        ]
        pending = pending_cve_ids(existing, language_ids, language, args.retry_failed)
        translations: dict[str, str] = {}
        generation_metadata: dict[str, dict[str, str]] = {}
        generation_started = time.perf_counter()
        for start in range(0, len(pending), args.batch_size):
            batch_ids = pending[start : start + args.batch_size]
            attempts = 0
            batch_started = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            while attempts <= args.max_retries:
                attempts += 1
                try:
                    protected_groups: list[tuple[list[str], dict[str, str]]] = []
                    layout_groups: list[list[list[_TranslationPart]]] = []
                    unit_counts: list[int] = []
                    flat_units: list[str] = []
                    for cve_id in batch_ids:
                        protected_text, replacements = protect_technical_identifiers(
                            records[cve_id].description
                        )
                        protected_chunks = _chunk_text(
                            protected_text, tokenizer, args.source_max_length
                        )
                        units, layouts = _prepare_translation_units(
                            protected_chunks, replacements
                        )
                        protected_groups.append((protected_chunks, replacements))
                        layout_groups.append(layouts)
                        unit_counts.append(len(units))
                        flat_units.extend(units)
                    flat_translations, flat_completion_flags = (
                        _translate_batch(
                            tokenizer, model, flat_units, LANGUAGE_CODES[language],
                            args.batch_size, args.max_length, args.num_beams,
                            source_max_length=args.source_max_length,
                            device=device,
                            return_completion=True,
                        )
                        if flat_units else ([], [])
                    )
                    offset = 0
                    for cve_id, (chunks, replacements), layouts, unit_count in zip(
                        batch_ids, protected_groups, layout_groups, unit_counts,
                        strict=True,
                    ):
                        translated_units = flat_translations[offset : offset + unit_count]
                        completion_flags = flat_completion_flags[
                            offset : offset + unit_count
                        ]
                        offset += unit_count
                        translated_protected = _render_translated_chunks(
                            layouts, translated_units
                        )
                        text, missing_placeholders = restore_technical_identifiers(
                            translated_protected, replacements
                        )
                        identifier_integrity = technical_identifier_occurrence_report(
                            records[cve_id].description, text
                        )
                        source_count = len(tokenizer(records[cve_id].description, add_special_tokens=True)["input_ids"])
                        output_count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
                        translations[cve_id] = text
                        generation_metadata[cve_id] = {
                            "source_token_count": str(source_count),
                            "source_truncated": "false",
                            "output_token_count": str(output_count),
                            "output_truncated": str(
                                _translation_units_truncated(
                                    translated_units, completion_flags
                                )
                            ).lower(),
                            "incomplete_units": str(
                                sum(not completed for completed in completion_flags)
                            ),
                            "chunk_count": str(len(chunks)),
                            "placeholder_count": str(len(replacements)),
                            "missing_placeholders": str(len(missing_placeholders)),
                            "placeholder_error": ", ".join(missing_placeholders),
                            "identifier_integrity_error": " | ".join(
                                identifier_integrity["issues"]
                            ),
                            "generation_attempts": str(attempts),
                            "error": "",
                        }
                    break
                except (RuntimeError, ValueError, IndexError) as exc:
                    if attempts > args.max_retries:
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        for cve_id in batch_ids:
                            translations[cve_id] = ""
                            generation_metadata[cve_id] = {
                                "source_token_count": "",
                                "source_truncated": "",
                                "output_token_count": "0",
                                "output_truncated": "",
                                "incomplete_units": "0",
                                "chunk_count": "0",
                                "placeholder_count": "0",
                                "missing_placeholders": "0",
                                "placeholder_error": "",
                                "identifier_integrity_error": "",
                                "generation_attempts": str(attempts),
                                "error": f"{type(exc).__name__}: {exc}",
                            }
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_seconds = time.perf_counter() - batch_started
            generated_pair_count += len(batch_ids)
            for cve_id in batch_ids:
                record = records[cve_id]
                text = translations.get(cve_id, "")
                metadata = generation_metadata.get(cve_id, {})
                status, identifiers = validate_translation_for_language(
                    record.description, text, language
                )
                missing_identifiers = translation_identifier_report(
                    record.description, text
                )["missing"]
                placeholder_error = metadata.get("placeholder_error", "")
                residual_placeholder = bool(
                    validate_technical_placeholders(text, {})
                )
                validation_errors = []
                if metadata.get("missing_placeholders", "0") != "0":
                    validation_errors.append(
                        f"Placeholder restoration failed: {placeholder_error}"
                    )
                if residual_placeholder:
                    validation_errors.append(
                        "Residual technical-identifier placeholder remains"
                    )
                if missing_identifiers:
                    validation_errors.append(
                        "Technical identifiers are missing after restoration"
                    )
                if metadata.get("identifier_integrity_error"):
                    validation_errors.append(
                        "Technical identifier integrity failed: "
                        + metadata["identifier_integrity_error"]
                    )
                if metadata.get("source_truncated") == "true":
                    validation_errors.append("Source text was truncated")
                if metadata.get("output_truncated") == "true":
                    validation_errors.append("Translation output was truncated")
                generation_error = metadata.get("error", "")
                if validation_errors or generation_error:
                    status = "failed"
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
                    "missing_identifiers": "|".join(missing_identifiers),
                    "source_token_count": metadata.get("source_token_count", ""),
                    "source_truncated": metadata.get("source_truncated", ""),
                    "output_token_count": metadata.get("output_token_count", "0"),
                    "output_truncated": metadata.get("output_truncated", ""),
                    "incomplete_units": metadata.get("incomplete_units", "0"),
                    "chunk_count": metadata.get("chunk_count", "0"),
                    "placeholder_count": metadata.get("placeholder_count", "0"),
                    "missing_placeholders": metadata.get("missing_placeholders", "0"),
                    "generation_attempts": metadata.get("generation_attempts", "0"),
                    "generation_seconds": f"{batch_seconds / len(batch_ids):.6f}",
                    "generated_at": datetime.now(timezone.utc).isoformat(),
                    "error": generation_error,
                    "validation_error": "; ".join(validation_errors),
                    "placeholder_error": placeholder_error,
                    "identifier_integrity_error": metadata.get(
                        "identifier_integrity_error", ""
                    ),
                }
                if first_translation is None and text:
                    first_translation = {
                        "cve_id": cve_id,
                        "language": language,
                        "output_characters": len(text),
                        "validation_status": status,
                        "generation_seconds": batch_seconds / len(batch_ids),
                        "device": str(device),
                    }
                if first_valid_translation is None and text and status == "valid":
                    first_valid_translation = {
                        "cve_id": cve_id,
                        "language": language,
                        "output_characters": len(text),
                        "validation_status": status,
                        "generation_seconds": batch_seconds / len(batch_ids),
                        "device": str(device),
                    }

            _write_rows(
                args.output,
                [
                    existing[key]
                    for key in sorted(existing)
                    if requested_pairs is None or key in requested_pairs
                ],
            )
        language_seconds = time.perf_counter() - generation_started
        generation_seconds_by_language[language] = language_seconds
        pairs_per_second_by_language[language] = (
            len(pending) / language_seconds if language_seconds else 0.0
        )

    final_rows = [
        row for key, row in existing.items()
        if requested_pairs is None or key in requested_pairs
    ]
    total_elapsed_seconds = time.perf_counter() - run_started
    cuda_peak_allocated_bytes = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    cuda_peak_reserved_bytes = (
        torch.cuda.max_memory_reserved(device) if device.type == "cuda" else 0
    )
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
            "max_new_tokens_ceiling": args.max_length,
            "per_unit_max_new_tokens": "min(ceiling, max(32, 2 * source_token_count))",
            "completion_check": "generated sequence contains tokenizer EOS token",
            "batch_size": args.batch_size,
            "source_max_length": args.source_max_length,
            "threads": args.threads,
            "max_retries": args.max_retries,
            "device": str(device),
            "cuda_used": device.type == "cuda" and generated_pair_count > 0,
            "local_files_only": args.local_files_only,
        },
        "profiling": {
            "model_load_seconds": model_load_seconds,
            "generation_seconds_by_language": generation_seconds_by_language,
            "pairs_per_second_by_language": pairs_per_second_by_language,
            "generated_pair_count": generated_pair_count,
            "total_elapsed_seconds": total_elapsed_seconds,
            "aggregate_pairs_per_second": (
                generated_pair_count / total_elapsed_seconds if total_elapsed_seconds else 0.0
            ),
            "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
            "cuda_peak_reserved_bytes": cuda_peak_reserved_bytes,
            "first_translation": first_translation,
            "first_valid_translation": first_valid_translation,
            "note": "Language timings include tokenization, model.generate, and batch checkpoint writes.",
        },
        "records_sha256": _sha256(args.records),
        "splits_sha256": _sha256(args.splits),
        "split_scope": args.split_scope,
        "selected_cve_ids": selected_ids,
        "test_cve_ids": selected_ids if args.split_scope == "test" else [],
        "requested_pair_count": len(requested_pairs)
        if requested_pairs is not None
        else len(selected_ids) * len(TRANSLATION_LANGUAGES),
        "requested_pairs_sha256": hashlib.sha256(
            "\n".join(
                f"{cve_id},{language}"
                for cve_id, language in sorted(
                    requested_pairs
                    if requested_pairs is not None
                    else {
                        (cve_id, language)
                        for cve_id in selected_ids
                        for language in TRANSLATION_LANGUAGES
                    }
                )
            ).encode("utf-8")
        ).hexdigest(),
        "row_count": len(final_rows),
        "counts_by_status": {
            status: Counter(row.get("validation_status", "") for row in final_rows)[status]
            for status in (
                "valid", "needs_review", "failed", "missing",
                "technical_identifier_missing", "suspiciously_short",
            )
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
    if args.gpu_report_output is not None:
        nvidia_smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,memory.used,memory.free",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        gpu_report = {
            "status": "verified" if device.type == "cuda" and generated_pair_count > 0 else "not_verified",
            "python_executable": sys.executable,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "device_total_memory_bytes": (
                torch.cuda.get_device_properties(device).total_memory
                if device.type == "cuda" else None
            ),
            "tensor_operation": tensor_check,
            "model": model_name,
            "model_revision": model_revision,
            "model_load_seconds_including_cuda_transfer": model_load_seconds,
            "actual_cuda_translation": first_translation,
            "validated_cuda_translation": first_valid_translation,
            "generation_seconds_by_language": generation_seconds_by_language,
            "generated_pair_count": generated_pair_count,
            "aggregate_pairs_per_second": (
                generated_pair_count / total_elapsed_seconds if total_elapsed_seconds else 0.0
            ),
            "counts_by_validation_status": manifest["counts_by_status"],
            "total_elapsed_seconds": total_elapsed_seconds,
            "cuda_peak_allocated_bytes": cuda_peak_allocated_bytes,
            "cuda_peak_reserved_bytes": cuda_peak_reserved_bytes,
            "nvidia_smi": {
                "returncode": nvidia_smi.returncode,
                "stdout": nvidia_smi.stdout.strip(),
                "stderr": nvidia_smi.stderr.strip(),
            },
            "dependency_versions": manifest["dependency_versions"],
            "errors": [
                row["error"] for row in final_rows if row.get("error")
            ],
        }
        args.gpu_report_output.parent.mkdir(parents=True, exist_ok=True)
        args.gpu_report_output.write_text(json.dumps(gpu_report, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
