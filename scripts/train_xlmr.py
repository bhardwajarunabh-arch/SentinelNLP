"""Explicit XLM-R fine-tuning entry point for a GPU-enabled environment."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train XLM-R in a GPU-enabled environment.")
    parser.add_argument("--data", required=True, help="JSON list of records with description and severity")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args()
    try:
        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "XLM-R training requires torch and transformers. Install them in Colab or a dedicated training environment."
        ) from exc

    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))
    labels = sorted({row["severity"] for row in rows})
    label_to_id = {label: index for index, label in enumerate(labels)}
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    class VulnerabilityDataset(Dataset):
        def __init__(self, records: list[dict[str, str]]) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            row = self.records[index]
            encoded = tokenizer(
                row["description"],
                truncation=True,
                padding="max_length",
                max_length=args.max_length,
                return_tensors="pt",
            )
            return {
                key: value.squeeze(0)
                for key, value in encoded.items()
            } | {"labels": torch.tensor(label_to_id[row["severity"]])}

    model = AutoModelForSequenceClassification.from_pretrained(
        "xlm-roberta-base", num_labels=len(labels), id2label=dict(enumerate(labels)), label2id=label_to_id
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output),
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            eval_strategy="no",
            save_strategy="epoch",
            logging_strategy="steps",
            report_to=[],
        ),
        train_dataset=VulnerabilityDataset(rows),
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output)
    tokenizer.save_pretrained(output)
    (output / "label_mapping.json").write_text(json.dumps(label_to_id, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
