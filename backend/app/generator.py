"""From-scratch decoder (Task C) — the 'G' in RAG.

Loads the trained DecoderLM and produces a grounded explanation from the review
+ retrieved neighbours, with NO external LLM. This is the *primary* generator;
`llm.generate_explanation` (Groq) is the fallback when this fails or is forced off.

Mirrors the inference loop in rag_agent.py, but takes neighbours from Supabase.
"""

import re
import sys
from functools import lru_cache

import torch

from .config import REPO_ROOT

# rag_common lives at the repo root (the original NLP pipeline).
sys.path.insert(0, str(REPO_ROOT))
import rag_common as C  # noqa: E402


def _polish(text: str, sentiment: str) -> str:
    """Tidy the raw decoder output into a clean sentence (drop <UNK>/specials,
    collapse spaces, capitalise, end with a period)."""
    text = re.sub(r"<[^>]+>", " ", text)          # strip <UNK>/<PAD>/etc.
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)   # no space before punctuation
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return f"This review reads as {sentiment.lower()} based on similar retrieved reviews."
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text += "."
    return text


class DecoderGenerator:
    def __init__(self):
        self.base_vocab, self.dec_vocab, self.inv_vocab = C.load_vocab()
        self.pad = self.base_vocab["<PAD>"]
        self.bos = self.dec_vocab["<BOS>"]
        self.eos = self.dec_vocab["<EOS>"]
        self.decoder = C.DecoderLM(vocab_size=len(self.dec_vocab), pad_idx=self.pad).to(C.DEVICE)
        self.decoder.load_state_dict(
            torch.load(REPO_ROOT / "models" / "decoder_best.pt", map_location=C.DEVICE)
        )
        self.decoder.eval()

    @torch.no_grad()
    def _decode(self, review: str, sent_id: int, len_id: int, neighbours: list[dict]) -> str:
        ctx_str = C.build_context_string(neighbours)
        src = C.build_input_sequence(review, sent_id, len_id, ctx_str, self.dec_vocab)
        gen = [self.bos]
        for _ in range(C.DEC_TGT_LEN):
            inp = torch.tensor([src + gen], dtype=torch.long, device=C.DEVICE)
            logits = self.decoder(inp, key_pad_mask=(inp == self.pad))
            nxt = int(logits[0, -1, :].argmax(-1).item())
            if nxt == self.eos:
                break
            gen.append(nxt)
        return " ".join(self.inv_vocab.get(t, "<UNK>") for t in gen[1:])

    def generate(self, review: str, sentiment: str, neighbours: list[dict]) -> str:
        """Grounded hybrid: the decoder generates the explanation, but the
        *evidence* slot is filled with the real top content words extracted from
        the retrieved neighbours (true retrieval grounding) — never the decoder's
        mode-collapsed guess. 100% from-scratch, no external LLM."""
        sent_id = C.SENTIMENT_MAP.get(sentiment, 1)
        n_words = len(str(review).split())
        len_id = 0 if n_words < 20 else (1 if n_words < 60 else 2)

        decoded = self._decode(review, sent_id, len_id, neighbours)

        # Real evidence: top content words across neighbours that aren't in the query.
        neighbour_texts = [str(n.get("text", "")) for n in neighbours]
        evidence = C.neighbour_evidence_words(neighbour_texts, review, k=4)
        ev_str = ", ".join(evidence) if evidence else "the retrieved similar reviews"
        sent_word = sentiment.lower()

        # Keep the decoder's own phrasing up to "mention", then ground the evidence.
        m = re.search(r"(.*\bmention\b)", decoded, re.IGNORECASE)
        if m and len(m.group(1).split()) >= 4:
            grounded = f"{m.group(1)} {ev_str}"
        else:
            grounded = f"This review is {sent_word} because similar reviewers mention {ev_str}"
        return _polish(grounded, sentiment)


@lru_cache
def get_generator() -> DecoderGenerator:
    return DecoderGenerator()
