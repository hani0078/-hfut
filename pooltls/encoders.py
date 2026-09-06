from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .text import batched, l2_normalize


class TextEncoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class LocalTextEncoder:
    """Frozen local Transformer encoder with attention-mask mean pooling.

    PoolTLS supplies the configured GTE-large directory via paths.gte_model.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda:0",
        batch_size: int = 32,
        max_length: int = 192,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._torch = torch
        self.device = device
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), local_files_only=True, use_fast=True
        )
        self.model = AutoModel.from_pretrained(
            str(model_path), local_files_only=True
        ).to(device)
        self.model.eval()
        self.model.requires_grad_(False)
        self._cache: dict[str, np.ndarray] = {}

    def _encode_missing(self, texts: Sequence[str]) -> None:
        torch = self._torch
        for chunk in batched(tuple(texts), self.batch_size):
            encoded = self.tokenizer(
                list(chunk),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            vectors = l2_normalize(pooled.float().cpu().numpy())
            for text, vector in zip(chunk, vectors, strict=True):
                self._cache[text] = vector

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = tuple(str(text) for text in texts)
        if not values:
            hidden_size = int(getattr(self.model.config, "hidden_size", 0))
            return np.empty((0, hidden_size), dtype=np.float32)
        missing = tuple(dict.fromkeys(text for text in values if text not in self._cache))
        if missing:
            self._encode_missing(missing)
        return np.stack([self._cache[text] for text in values]).astype(np.float32)
