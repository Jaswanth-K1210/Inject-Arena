"""Embedding cache for stealth + novelty rewards.

Wraps ``sentence-transformers/all-MiniLM-L6-v2`` (~80MB, runs fine on
Mac CPU) and pre-computes an embedding for every benign reference in
``scenarios/benign_refs.jsonl``. The reward function uses these to
score how closely a candidate payload resembles the benign distribution
of its slot ("stealth") and how different it is from recent attacker
outputs ("novelty").

The model and reference embeddings are loaded lazily on first use so
unit tests that don't need them avoid the 80MB download.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_REFS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "scenarios" / "benign_refs.jsonl"
)


class EmbeddingCache:
    """Lazy-loaded sentence-transformer with per-channel benign references."""

    def __init__(
        self,
        refs_path: Path | str = DEFAULT_REFS_PATH,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self.refs_path = Path(refs_path)
        self.model_name = model_name
        self._model = None  # loaded on first .encode() call
        self._channel_refs: Dict[str, List[str]] = self._load_refs(self.refs_path)
        self._channel_vecs: Dict[str, np.ndarray] = {}

    @staticmethod
    def _load_refs(path: Path) -> Dict[str, List[str]]:
        if not path.exists():
            raise FileNotFoundError(f"benign_refs not found at {path}")
        out: Dict[str, List[str]] = {}
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out.setdefault(rec["channel"], []).append(rec["text"])
        return out

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)

    def _ensure_channel_vecs(self, channel: str) -> np.ndarray:
        if channel in self._channel_vecs:
            return self._channel_vecs[channel]
        if channel not in self._channel_refs:
            raise KeyError(f"no benign refs for channel {channel!r}")
        self._ensure_model()
        vecs = self._model.encode(  # type: ignore[union-attr]
            self._channel_refs[channel],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vecs = np.asarray(vecs, dtype=np.float32)
        self._channel_vecs[channel] = vecs
        return vecs

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_model()
        v = self._model.encode(  # type: ignore[union-attr]
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(v, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public scoring API
    # ------------------------------------------------------------------

    def stealth_score(self, payload: str, channel: str) -> float:
        """Max cosine similarity between payload and that channel's benign refs.

        Higher = the payload looks more like benign content for this slot.
        """
        if not payload or not payload.strip():
            return 0.0
        refs = self._ensure_channel_vecs(channel)
        emb = self._encode([payload])[0]
        sims = refs @ emb  # cosine since both sides are unit-normalized
        return float(np.clip(sims.max(), 0.0, 1.0))

    def novelty_score(self, payload: str, recent_payloads: Sequence[str]) -> float:
        """1 - max cosine similarity between payload and any recent payload.

        Higher = more novel. Empty ``recent_payloads`` -> 1.0 (max novelty).
        """
        if not recent_payloads:
            return 1.0
        if not payload or not payload.strip():
            return 0.0
        all_texts = [payload, *recent_payloads]
        vecs = self._encode(all_texts)
        sims = vecs[0] @ vecs[1:].T
        max_sim = float(np.clip(sims.max(), 0.0, 1.0))
        return float(np.clip(1.0 - max_sim, 0.0, 1.0))
