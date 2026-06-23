"""Embedding engine — serves the project's own Task-A Transformer encoder.

Produces 128-dim L2-normalised vectors so retrieval can use cosine distance in
Supabase pgvector. Loaded once as a process-wide singleton.
"""

import sys
from functools import lru_cache

import numpy as np
import torch

from .config import get_settings, REPO_ROOT

# rag_common lives at the repo root (the original NLP pipeline).
sys.path.insert(0, str(REPO_ROOT))
import rag_common as C  # noqa: E402


class Embedder:
    def __init__(self):
        s = get_settings()
        self.base_vocab, _, _ = C.load_vocab()
        self.pad = self.base_vocab["<PAD>"]
        self.encoder = C.EncoderModel(vocab_size=len(self.base_vocab), pad_idx=self.pad).to(C.DEVICE)
        self.encoder.load_state_dict(torch.load(s.encoder_path, map_location=C.DEVICE))
        self.encoder.eval()
        self.dim = C.EMBED_DIM

    def _ids(self, text: str):
        ids = C.tokenize(text, self.base_vocab, C.ENC_MAX_LEN)
        return ids + [self.pad] * (C.ENC_MAX_LEN - len(ids))

    @torch.no_grad()
    def embed(self, texts: list[str], batch_size: int = 128) -> np.ndarray:
        out = []
        for s in range(0, len(texts), batch_size):
            chunk = texts[s:s + batch_size]
            t = torch.tensor([self._ids(x) for x in chunk], dtype=torch.long, device=C.DEVICE)
            _, _, emb = self.encoder(t)
            out.append(emb.cpu().numpy())
        v = np.concatenate(out, axis=0)
        v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        return v.astype(np.float32)

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0].tolist()

    @torch.no_grad()
    def predict_sentiment(self, text: str) -> str:
        t = torch.tensor([self._ids(text)], dtype=torch.long, device=C.DEVICE)
        s_logits, _, _ = self.encoder(t)
        return ["Negative", "Neutral", "Positive"][int(s_logits.argmax(-1).item())]


@lru_cache
def get_embedder() -> Embedder:
    return Embedder()
