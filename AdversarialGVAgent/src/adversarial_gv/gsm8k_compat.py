"""Compatibility copy of TextGrad's GSM8K_DSPy split loader.

TextGrad 0.1.8 uses the legacy Hugging Face id ``gsm8k``. Current versions of
``huggingface_hub`` require the namespaced id ``openai/gsm8k``. This module
preserves TextGrad's normalization, shuffle seeds, and split sizes without
modifying the original project.
"""

import random
from typing import Dict, List, Optional


class GSM8KDSPyCompat:
    def __init__(self, root: Optional[str] = None, split: str = "train"):
        from datasets import load_dataset

        if split not in {"train", "val", "test"}:
            raise ValueError(f"unsupported GSM8K split: {split}")
        dataset = load_dataset("openai/gsm8k", "main", cache_dir=root)
        official_train = self._normalize(dataset["train"])
        official_test = self._normalize(dataset["test"])
        random.Random(0).shuffle(official_train)
        random.Random(0).shuffle(official_test)
        splits = {
            "train": official_train[:200],
            "val": official_train[200:500],
            "test": official_test,
        }
        self.data = splits[split]
        self.split = split

    @staticmethod
    def _normalize(rows) -> List[Dict[str, str]]:
        normalized = []
        for example in rows:
            answer_parts = example["answer"].strip().split()
            if len(answer_parts) < 2 or answer_parts[-2] != "####":
                raise ValueError("unexpected GSM8K answer format")
            normalized.append(
                {
                    "question": example["question"],
                    "gold_reasoning": " ".join(answer_parts[:-2]),
                    "answer": str(int(answer_parts[-1].replace(",", ""))),
                }
            )
        return normalized

    def __getitem__(self, index: int):
        row = self.data[index]
        return f"Question: {row['question']}", row["answer"], row["gold_reasoning"]

    def __len__(self) -> int:
        return len(self.data)
