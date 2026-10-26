from pathlib import Path

import pytest

from backend.app.transformer_inference import TransformerInferenceService


def test_transformer_service_reports_missing_model(tmp_path: Path) -> None:
    service = TransformerInferenceService(tmp_path / "missing")

    service.load()

    assert not service.available
    assert service.load_error == f"Transformer model directory does not exist: {tmp_path / 'missing'}"
    with pytest.raises(RuntimeError, match="does not exist"):
        service.predict("test text")


def test_transformer_service_is_unavailable_until_load_succeeds(tmp_path: Path) -> None:
    service = TransformerInferenceService(tmp_path / "model")
    service._tokenizer = object()
    service._model = object()
    service._torch = object()

    assert not service.available
