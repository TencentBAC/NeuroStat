"""Datasets, collators and sample builders for NeuroStat.

Two JSON data formats are supported:

- ``"MIRAGE"``: ``[{"original": str, "rewritten": str}, ...]``
- ``"pair"``:   ``{"original": [str, ...], "rewritten": [str, ...]}``

In both cases ``original`` = human-written, ``rewritten`` = machine-generated.
"""

from __future__ import annotations

import json
import random

import torch
from torch.utils.data import Dataset


LABEL_TO_ID = {"Human": 0, "AIGC": 1}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


# ---------------------------------------------------------------------------
# Evaluation-time dataset (yields raw pair dicts, scoring is done by the model)
# ---------------------------------------------------------------------------

class PairTextDataset(Dataset):
    """Yields one ``{original, rewritten}`` pair per item."""

    def __init__(self, data_path: str, data_format: str = "MIRAGE"):
        super().__init__()
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.data_format = data_format

    def __getitem__(self, index):
        if self.data_format == "MIRAGE":
            return {
                "original": self.data[index]["original"],
                "rewritten": self.data[index]["rewritten"],
            }
        return {
            "original": self.data["original"][index],
            "rewritten": self.data["rewritten"][index],
        }

    def __len__(self):
        if self.data_format == "MIRAGE":
            return len(self.data)
        return len(self.data["original"])

    @staticmethod
    def collate_fn(batch):
        return {
            "original": [item["original"] for item in batch],
            "rewritten": [item["rewritten"] for item in batch],
        }


# ---------------------------------------------------------------------------
# Training-time binary classification dataset
# ---------------------------------------------------------------------------

class BinaryTextDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)


class SequenceClassificationCollator:
    def __init__(self, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features: list[dict]):
        texts = [f["text"] for f in features]
        labels = [f["labels"] for f in features]
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def load_pair_data(data_path: str, data_format: str = "MIRAGE") -> tuple[list[str], list[str]]:
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data_format == "MIRAGE":
        original_texts = [item["original"] for item in data]
        rewritten_texts = [item["rewritten"] for item in data]
    elif data_format == "pair":
        original_texts = data["original"]
        rewritten_texts = data["rewritten"]
    else:
        raise ValueError(
            f"Unsupported data_format: {data_format!r}. Expected 'MIRAGE' or 'pair'."
        )

    if len(original_texts) != len(rewritten_texts):
        raise ValueError("Number of original texts does not match rewritten texts.")

    return original_texts, rewritten_texts


def build_binary_classification_samples(
    data_path: str,
    data_format: str = "MIRAGE",
    shuffle: bool = True,
    seed: int = 42,
) -> list[dict]:
    original_texts, rewritten_texts = load_pair_data(data_path, data_format)
    samples = [{"text": t, "labels": LABEL_TO_ID["Human"]} for t in original_texts]
    samples.extend({"text": t, "labels": LABEL_TO_ID["AIGC"]} for t in rewritten_texts)
    if shuffle:
        random.Random(seed).shuffle(samples)
    return samples
