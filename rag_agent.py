"""
rag_agent.py — Deployable Retrieval-Augmented Sentiment-Explanation Agent
=========================================================================
The end-to-end product: given a single raw review, the agent runs the full RAG
loop and returns a grounded explanation of its sentiment.

    review ──▶ ENCODE ──▶ RETRIEVE k similar reviews ──▶ GENERATE explanation
              (Task A)      (Task B, cosine search)        (Task C decoder)

The explanation is grounded in evidence drawn from the retrieved neighbours, so
it cites what *similar customers* said — not just the review itself.

Usage
-----
    # one review from the command line
    python rag_agent.py --review "This thing broke after two days, total waste of money."

    # interactive prompt loop
    python rag_agent.py

    # JSON output (for piping into a service / demo)
    python rag_agent.py --review "Best purchase I've made all year!" --json

Programmatic
------------
    from rag_agent import RagAgent
    agent = RagAgent()
    result = agent.explain("Battery died in a week.")
    print(result["explanation"])
"""

import argparse
import json
import sys

import numpy as np
import pandas as pd
import torch

import rag_common as C


class RagAgent:
    """Loads the trained encoder + decoder and the retrieval index once, then
    serves explanations for arbitrary review text."""

    def __init__(self, top_k=5):
        self.top_k = top_k
        self.base_vocab, self.dec_vocab, self.inv_vocab = C.load_vocab()
        self.pad = self.base_vocab["<PAD>"]
        self.bos = self.dec_vocab["<BOS>"]
        self.eos = self.dec_vocab["<EOS>"]

        # Encoder (Task A) — produces the query embedding for retrieval.
        self.encoder = C.EncoderModel(vocab_size=len(self.base_vocab), pad_idx=self.pad).to(C.DEVICE)
        self.encoder.load_state_dict(torch.load(C.MDL_DIR / "encoder_best.pt", map_location=C.DEVICE))
        self.encoder.eval()

        # Decoder (Task C) — generates the grounded explanation.
        self.decoder = C.DecoderLM(vocab_size=len(self.dec_vocab), pad_idx=self.pad).to(C.DEVICE)
        self.decoder.load_state_dict(torch.load(C.MDL_DIR / "decoder_best.pt", map_location=C.DEVICE))
        self.decoder.eval()

        # Retrieval index (Task B) — the searchable corpus of training reviews.
        db = np.load(C.RES_DIR / "train_embeddings.npy")
        self.db_norm = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-9)
        self.train_df = pd.read_csv(C.DATA_DIR / "train.csv")

    # ── pipeline stages ───────────────────────────────────────────────────────
    def _encode(self, text):
        ids = C.tokenize(text, self.base_vocab, C.ENC_MAX_LEN)
        ids = ids + [self.pad] * (C.ENC_MAX_LEN - len(ids))
        with torch.no_grad():
            s_logits, _, emb = self.encoder(torch.tensor([ids], dtype=torch.long, device=C.DEVICE))
        v = emb.squeeze(0).cpu().numpy()
        v /= (np.linalg.norm(v) + 1e-9)
        pred_sent = ["Negative", "Neutral", "Positive"][int(s_logits.argmax(-1).item())]
        return v, pred_sent

    def _retrieve(self, qvec):
        sims = self.db_norm @ qvec
        idx = np.argsort(-sims)[: self.top_k]
        return [
            {"text": str(self.train_df.iloc[j]["text"]),
             "sentiment": self.train_df.iloc[j]["sentiment"],
             "score": float(sims[j])}
            for j in idx
        ]

    def _generate(self, review, sentiment, neighbours):
        sent_id = C.SENTIMENT_MAP[sentiment]
        # length bucket, mirroring clean_data.py
        n = len(str(review).split())
        len_id = 0 if n < 20 else (1 if n < 60 else 2)
        ctx_str = C.build_context_string(neighbours)
        src = C.build_input_sequence(review, sent_id, len_id, ctx_str, self.dec_vocab)

        gen = [self.bos]
        with torch.no_grad():
            for _ in range(C.DEC_TGT_LEN):
                inp = torch.tensor([src + gen], dtype=torch.long, device=C.DEVICE)
                logits = self.decoder(inp, key_pad_mask=(inp == self.pad))
                nxt = logits[0, -1, :].argmax(-1).item()
                if nxt == self.eos:
                    break
                gen.append(nxt)
        return " ".join(self.inv_vocab.get(t, "<UNK>") for t in gen[1:])

    # ── public API ────────────────────────────────────────────────────────────
    def explain(self, review, override_sentiment=None):
        qvec, pred_sent = self._encode(review)
        sentiment = override_sentiment or pred_sent
        neighbours = self._retrieve(qvec)
        explanation = self._generate(review, sentiment, neighbours)
        return {
            "review": review,
            "predicted_sentiment": pred_sent,
            "retrieved": neighbours,
            "explanation": explanation,
        }


# ──────────────────────────────────────────────────────────────────────────────
# CLI presentation
# ──────────────────────────────────────────────────────────────────────────────
def _print_human(res):
    print("\n" + "=" * 74)
    print("REVIEW")
    print("-" * 74)
    print(f"  {res['review']}")
    print(f"\nPREDICTED SENTIMENT:  {res['predicted_sentiment']}")
    print("\nRETRIEVED SIMILAR REVIEWS  (the 'R' in RAG)")
    print("-" * 74)
    for i, n in enumerate(res["retrieved"], 1):
        print(f"  {i}. ({n['sentiment']}, sim={n['score']:.3f}) {n['text'][:90]}")
    print("\nGROUNDED EXPLANATION  (the 'AG' in RAG)")
    print("-" * 74)
    print(f"  {res['explanation']}")
    print("=" * 74 + "\n")


def main():
    ap = argparse.ArgumentParser(description="Retrieval-Augmented sentiment-explanation agent")
    ap.add_argument("--review", type=str, help="review text to explain")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of formatted text")
    args = ap.parse_args()

    print("Loading RAG agent (encoder + retrieval index + decoder)...", file=sys.stderr)
    agent = RagAgent(top_k=args.top_k)

    if args.review:
        res = agent.explain(args.review)
        print(json.dumps(res, indent=2) if args.json else "", end="")
        if not args.json:
            _print_human(res)
        return

    print("Interactive mode — type a review, or 'quit' to exit.", file=sys.stderr)
    while True:
        try:
            text = input("\nreview> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit", "q"}:
            break
        if text:
            _print_human(agent.explain(text))


if __name__ == "__main__":
    main()
