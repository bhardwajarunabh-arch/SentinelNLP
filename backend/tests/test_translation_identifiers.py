from backend.app.translations import (
    protect_technical_identifiers,
    restore_technical_identifiers,
    technical_identifiers,
    translation_identifier_report,
    validate_translation_for_language,
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
