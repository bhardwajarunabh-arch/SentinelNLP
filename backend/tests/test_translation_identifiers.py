from backend.app.translations import (
    protect_technical_identifiers,
    restore_technical_identifiers,
    technical_identifiers,
    technical_identifier_occurrence_report,
    translation_identifier_report,
    validate_technical_placeholders,
    validate_translation_for_language,
)
import torch

from scripts.translate_expanded_test import (
    _translate_batch,
    _per_unit_generation_cap,
    _prepare_translation_units,
    _render_translated_chunks,
    _translation_units_truncated,
)


SOURCE = (
    "CVE-2024-12345 affects Product_X 1.2.3 in /opt/app/config.yaml; "
    "the fix is in abcdef0123456789."
)


def test_identifier_matching_ignores_sentence_punctuation() -> None:
    translated = (
        "CVE-2024-12345 afecta Product_X 1.2.3 en /opt/app/config.yaml; "
        "la corrección está en abcdef0123456789."
    )

    report = translation_identifier_report(SOURCE, translated)

    assert report["missing"] == []
    assert validate_translation_for_language(SOURCE, translated, "es")[0] == "valid"


def test_identifier_report_covers_security_relevant_categories() -> None:
    report = translation_identifier_report(SOURCE, SOURCE)

    assert set(report["required_categories"]) >= {
        "cve_id",
        "version",
        "hash",
        "path_or_filename",
        "symbol_or_product",
    }
    assert set(technical_identifiers(SOURCE)) == {
        "1.2.3",
        "CVE-2024-12345",
        "Product_X",
        "abcdef0123456789",
        "/opt/app/config.yaml",
    }


def test_angle_bracket_log_path_tokens_are_protected_per_occurrence() -> None:
    source = (
        "URI: /fpc/<fpc>/pfe/<pfe>/cm/<cm>/Host_Loopback/"
        "<cm>/HOST_LOOPBACK_MAKE_CMERROR_ID[<id>]"
    )

    protected, replacements = protect_technical_identifiers(source)
    restored, issues = restore_technical_identifiers(protected, replacements)
    report = technical_identifier_occurrence_report(source, restored)

    assert list(replacements.values()).count("<cm>") == 2
    assert {"<fpc>", "<pfe>", "<cm>", "<id>"} <= set(replacements.values())
    assert issues == []
    assert restored == source
    assert report["issues"] == []


def test_missing_translation_is_not_counted_as_valid() -> None:
    status, missing = validate_translation_for_language(SOURCE, "", "hi")

    assert status == "missing"
    assert missing == []


def test_identifier_placeholders_restore_exact_values() -> None:
    protected, replacements = protect_technical_identifiers(SOURCE)

    restored, missing = restore_technical_identifiers(protected, replacements)

    assert missing == []
    assert restored == SOURCE
    assert all(identifier not in protected for identifier in replacements.values())


def test_identifier_placeholder_loss_is_reported() -> None:
    protected, replacements = protect_technical_identifiers(SOURCE)
    first_placeholder = next(iter(replacements))

    restored, missing = restore_technical_identifiers(
        protected.replace(first_placeholder, ""),
        replacements,
    )

    assert first_placeholder not in restored
    assert missing == [replacements[first_placeholder]]


def test_repeated_identifiers_receive_distinct_placeholders() -> None:
    source = "CVE-2024-12345 affects package-x 1.2.3; CVE-2024-12345 affects package-x 1.2.3."

    protected, replacements = protect_technical_identifiers(source)
    restored, issues = restore_technical_identifiers(protected, replacements)

    assert len(replacements) == 4
    assert list(replacements.values()).count("CVE-2024-12345") == 2
    assert list(replacements.values()).count("1.2.3") == 2
    assert all(protected.count(placeholder) == 1 for placeholder in replacements)
    assert issues == []
    assert restored == source


def test_identifier_protection_covers_adjacent_punctuated_technical_values() -> None:
    source = "CVE-2024-0001,Widget.run()/opt/app.py; pkg_name-2.1.0! abcdef0123456789"

    protected, replacements = protect_technical_identifiers(source)
    restored, issues = restore_technical_identifiers(protected, replacements)

    assert {
        "CVE-2024-0001",
        "Widget.run",
        "/opt/app.py",
        "pkg_name-2.1.0",
        "abcdef0123456789",
    } <= set(replacements.values())
    assert issues == []
    assert restored == source


def test_unknown_and_duplicated_placeholders_are_rejected() -> None:
    protected, replacements = protect_technical_identifiers(SOURCE)
    marker = next(iter(replacements))

    duplicated, duplicate_issues = restore_technical_identifiers(
        protected + " " + marker, replacements
    )
    unknown_text = protected + " <SENTINEL_ID_999>"
    unknown, unknown_issues = restore_technical_identifiers(
        unknown_text, replacements
    )

    assert duplicated == protected + " " + marker
    assert any(
        "Duplicated placeholder" in issue
        for issue in validate_technical_placeholders(duplicated, replacements)
    )
    assert duplicate_issues == [replacements[marker]]
    assert unknown == unknown_text
    assert any(
        "Unknown placeholder" in issue
        for issue in validate_technical_placeholders(unknown_text, replacements)
    )
    assert any(issue.startswith("Unknown placeholder:") for issue in unknown_issues)


def test_placeholder_split_at_chunk_boundary_is_rejected() -> None:
    protected, replacements = protect_technical_identifiers(SOURCE)
    marker = next(iter(replacements))
    split_at = protected.index(marker) + len(marker) // 2
    chunks = [protected[:split_at], protected[split_at:]]

    assert validate_technical_placeholders(chunks, replacements)
    try:
        _prepare_translation_units(chunks, replacements)
    except ValueError as exc:
        assert "damaged by chunking" in str(exc)
    else:
        raise AssertionError("A placeholder split across chunks must be rejected")


def test_structured_placeholder_slots_restore_without_changing_other_text() -> None:
    protected, replacements = protect_technical_identifiers(SOURCE)
    units, layouts = _prepare_translation_units([protected], replacements)
    translated_units = [f"[{unit}]" for unit in units]
    reconstructed = _render_translated_chunks(layouts, translated_units)
    restored, issues = restore_technical_identifiers(reconstructed, replacements)

    assert issues == []
    assert "SENTINEL_ID_" not in restored
    assert all(f"[{part.strip()}]" in restored for part in units if part.strip())
    assert all(identifier in restored for identifier in replacements.values())


def test_span_boundary_punctuation_prevents_spanish_and_french_identifier_fusion() -> None:
    source = "/opt/app/config.yaml. The issue is in Widget.run."
    protected, replacements = protect_technical_identifiers(source)
    units, layouts = _prepare_translation_units([protected], replacements)

    assert units == ["The issue is in"]
    for translated_span in ("El problema está en", "Le problème est dans"):
        reconstructed = _render_translated_chunks(layouts, [translated_span])
        restored, issues = restore_technical_identifiers(
            reconstructed, replacements
        )

        assert issues == []
        assert restored.startswith("/opt/app/config.yaml. ")
        assert restored.endswith(" Widget.run.")
        assert technical_identifier_occurrence_report(source, restored)["issues"] == []


def test_chunk_boundary_punctuation_and_identifier_slots_reconstruct_exactly() -> None:
    source = "CVE-2024-0001. The issue is in Widget.run."
    protected, replacements = protect_technical_identifiers(source)
    marker = next(
        placeholder
        for placeholder, identifier in replacements.items()
        if identifier == "CVE-2024-0001"
    )
    split_at = protected.index(marker) + len(marker) + 2
    chunks = [protected[:split_at], protected[split_at:]]
    units, layouts = _prepare_translation_units(chunks, replacements)

    assert units == ["The issue is in"]
    translated = _render_translated_chunks(layouts, ["El problema está en"])
    restored, issues = restore_technical_identifiers(translated, replacements)

    assert issues == []
    assert restored == "CVE-2024-0001. El problema está en Widget.run."
    assert technical_identifier_occurrence_report(source, restored)["issues"] == []


def test_empty_generated_fragment_is_rejected() -> None:
    source = "CVE-2024-0001. The issue is in Widget.run."
    protected, replacements = protect_technical_identifiers(source)
    _, layouts = _prepare_translation_units([protected], replacements)

    try:
        _render_translated_chunks(layouts, [""])
    except ValueError as exc:
        assert "empty translation fragment" in str(exc)
    else:
        raise AssertionError("An empty model fragment must be rejected")


def test_nontechnical_na_text_is_not_protected_as_a_path() -> None:
    assert "n/a" not in technical_identifiers("The affected version is n/a.")


def test_truncation_is_measured_per_generated_unit_not_aggregate_output() -> None:
    assert not _translation_units_truncated(
        [" ".join(["first"] * 80), " ".join(["second"] * 80)],
        [True, True],
    )
    assert _translation_units_truncated(["incomplete"], [False])
    try:
        _translation_units_truncated(["one", "two"], [True])
    except ValueError as exc:
        assert "Every generated text unit" in str(exc)
    else:
        raise AssertionError("Completion must be recorded for each unit")


def test_generation_cap_is_computed_for_each_text_unit() -> None:
    assert _per_unit_generation_cap(8, 256) == 32
    assert _per_unit_generation_cap(80, 256) == 160
    assert _per_unit_generation_cap(200, 256) == 256


def test_translation_batch_uses_eos_to_report_per_unit_completion() -> None:
    class FakeTokenizer:
        eos_token_id = 2
        pad_token_id = 1

        def __call__(self, texts, **kwargs):
            values = [1 if text == "complete" else 0 for text in texts]
            return {
                "input_ids": torch.tensor([[value] for value in values]),
                "attention_mask": torch.ones((len(values), 1), dtype=torch.long),
            }

        def convert_tokens_to_ids(self, token):
            return 3

        def batch_decode(self, sequences, skip_special_tokens=True):
            return ["completed" if 2 in sequence.tolist() else "cut" for sequence in sequences]

    class FakeModel:
        def generate(self, input_ids, max_new_tokens, **kwargs):
            outputs = []
            for input_id in input_ids[:, 0].tolist():
                if input_id:
                    outputs.append([0, 2] + [1] * (max_new_tokens - 1))
                else:
                    outputs.append([0] + [4] * max_new_tokens)
            return torch.tensor(outputs)

    texts, completed = _translate_batch(
        FakeTokenizer(),
        FakeModel(),
        ["complete", "truncated"],
        "spa_Latn",
        batch_size=2,
        max_length=64,
        num_beams=1,
        return_completion=True,
    )

    assert texts == ["completed", "cut"]
    assert completed == [True, False]


def test_chinese_shortness_check_counts_han_characters_not_spaces() -> None:
    source = "Memory corruption in HLOS while running playready use-case."
    translated = "在运行时，HLOS内存损坏。"

    assert validate_translation_for_language(source, translated, "zh")[0] == "valid"


def test_identifier_occurrence_report_does_not_count_version_prefixes() -> None:
    source = "Version 16.1 before 16.1.6; PTX1000, PTX10002-60C."

    report = technical_identifier_occurrence_report(source, source)

    assert report["issues"] == []
    assert report["expected_occurrences"]["16.1"] == 1
    assert report["actual_occurrences"]["16.1"] == 1


def test_identifier_occurrence_report_rejects_duplicate_and_altered_values() -> None:
    source = "CVE-2024-12345 affects package-x 1.2.3."
    duplicated = technical_identifier_occurrence_report(
        source, "CVE-2024-12345 affects package-x 1.2.3 and 1.2.3."
    )
    altered = technical_identifier_occurrence_report(
        source, "CVE-2024-12345 affects package-x 1.2.4."
    )

    assert duplicated["duplicated"] == {"1.2.3": 1}
    assert altered["missing"] == {"1.2.3": 1}
    assert altered["unexpected"] == ["1.2.4"]
