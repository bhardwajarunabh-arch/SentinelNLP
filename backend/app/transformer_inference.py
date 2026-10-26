from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TransformerInferenceService:
    model_path: Path
    _tokenizer: Any = None
    _model: Any = None
    _torch: Any = None
    _loaded: bool = False
    load_error: str | None = None

    @property
    def available(self) -> bool:
        return (
            self._loaded
            and self._tokenizer is not None
            and self._model is not None
            and self._torch is not None
        )

    def load(self) -> None:
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._loaded = False
        self.load_error = None

        if not self.model_path.is_dir():
            self.load_error = f"Transformer model directory does not exist: {self.model_path}"
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            self.load_error = f"Transformer dependencies are unavailable: {exc}"
            return

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(self.model_path), local_files_only=True
            )
            self._model.eval()
            self._torch = torch
            self._loaded = True
        except (OSError, RuntimeError, ValueError) as exc:
            self._tokenizer = None
            self._model = None
            self._torch = None
            self.load_error = f"Transformer model could not be loaded: {exc}"

    def predict(self, text: str) -> tuple[str, dict[str, float]]:
        if not self.available:
            detail = self.load_error or "Transformer model is not loaded"
            raise RuntimeError(detail)

        inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
            probabilities = self._torch.softmax(logits, dim=-1)[0].tolist()

        id2label = self._model.config.id2label
        probability_map = {
            id2label[index]: float(probability)
            for index, probability in enumerate(probabilities)
        }
        predicted_index = int(self._torch.argmax(logits, dim=-1)[0])
        return id2label[predicted_index], probability_map
