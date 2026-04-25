"""Llama Prompt Guard 2 (86M) — input classifier defense.

Runs every attacker payload through a text-classification pipeline and
returns a DefenseVerdict. The model loads lazily on first .scan() call
and stays in GPU memory for the lifetime of the process.

Model card: https://huggingface.co/meta-llama/Llama-Prompt-Guard-2-86M
Labels: "BENIGN" | "MALICIOUS"
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import DefenseVerdict

logger = logging.getLogger(__name__)

# Model id for the 86M variant. There is also a 22M (faster) variant;
# we use 86M for quality — 92ms on A100 per model card.
_MODEL_ID = "meta-llama/Llama-Prompt-Guard-2-86M"

# PG2 has a 512-token context window. For longer payloads we scan the
# first 512 tokens (truncate=True in the pipeline handles this).
_MAX_LENGTH = 512


class PromptGuard:
    """Wrapper around Llama Prompt Guard 2."""

    name = "prompt_guard_2"

    def __init__(self, model_id: str = _MODEL_ID, device: Optional[str] = None) -> None:
        self._model_id = model_id
        self._device = device  # None → auto (GPU if available)
        self._pipeline = None

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline

            logger.info("Loading Llama Prompt Guard 2 (%s) …", self._model_id)
            self._pipeline = pipeline(
                "text-classification",
                model=self._model_id,
                device=self._device,
                truncation=True,
                max_length=_MAX_LENGTH,
            )
            logger.info("PromptGuard loaded.")
        except Exception as exc:
            raise RuntimeError(f"Failed to load PromptGuard model {self._model_id!r}: {exc}") from exc

    def scan(self, text: str) -> DefenseVerdict:
        """Return flagged=True if the text is classified as MALICIOUS."""
        self._ensure_loaded()

        if not text or not text.strip():
            return DefenseVerdict(flagged=False, score=0.0, reason="empty_input")

        result = self._pipeline(text)[0]  # type: ignore[index]
        label: str = result["label"].upper()
        score: float = float(result["score"])

        flagged = label == "MALICIOUS"
        pg2_score = score if flagged else 1.0 - score  # probability of being injection

        return DefenseVerdict(
            flagged=flagged,
            score=pg2_score,
            reason=f"pg2:{label.lower()}",
        )
