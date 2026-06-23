"""
Task B — Retrieval & Embeddings  (FIXED)
========================================
Builds the retrieval side of the RAG pipeline on top of the Task-A encoder.

What changed vs. the original:
  * Retrieves contexts for TRAIN and VAL too (not just TEST) so the decoder in
    Task C can actually be *trained* with real retrieved context — the original
    trained on empty dummy contexts, which is why "RAG" never helped.
  * Excludes self-retrieval when a train review queries the train index
    (otherwise every train query trivially retrieves itself at sim=1.0).
  * Vectorised cosine search (one matmul per chunk) instead of a Python loop.

Reuses the existing encoder_best.pt and results/train_embeddings.npy — the
encoder does not need retraining.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

import rag_common as C

# Subsample sizes — kept small so the whole pipeline retrains on CPU in minutes.
# Task_C reads the SAME first-N rows, so the two stay aligned.
TRAIN_N = 8000
VAL_N   = 1500
TEST_N  = 1500
TOP_K   = 5

base_vocab, dec_vocab, _ = C.load_vocab()
PAD_IDX = base_vocab["<PAD>"]


def load_encoder():
    enc = C.EncoderModel(vocab_size=len(base_vocab), pad_idx=PAD_IDX).to(C.DEVICE)
    enc.load_state_dict(torch.load(C.MDL_DIR / "encoder_best.pt", map_location=C.DEVICE))
    enc.eval()
    return enc


def encode_texts(texts, encoder, batch=256):
    """Encode a list of raw texts → L2-normalised CLS embeddings (N, 128)."""
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), batch):
            chunk = texts[s:s + batch]
            ids = [C.tokenize(t, base_vocab, C.ENC_MAX_LEN) for t in chunk]
            ids = [seq + [PAD_IDX] * (C.ENC_MAX_LEN - len(seq)) for seq in ids]
            t = torch.tensor(ids, dtype=torch.long, device=C.DEVICE)
            _, _, emb = encoder(t)
            out.append(emb.cpu().numpy())
    v = np.concatenate(out, axis=0)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return v


def topk_retrieve(query_vecs, db_norm, k, self_global_idx=None, chunk=512):
    """Cosine top-k via chunked matmul. self_global_idx[i] (if given) is masked
    out of row i's candidates to prevent self-retrieval on the train split."""
    N = query_vecs.shape[0]
    all_idx, all_sim = [], []
    for s in range(0, N, chunk):
        q = query_vecs[s:s + chunk]                       # (c, d)
        sims = q @ db_norm.T                              # (c, |DB|)
        if self_global_idx is not None:
            for r, gi in enumerate(self_global_idx[s:s + chunk]):
                sims[r, gi] = -np.inf
        idx = np.argpartition(-sims, kth=k, axis=1)[:, :k]
        row = np.arange(idx.shape[0])[:, None]
        order = np.argsort(-sims[row, idx], axis=1)       # sort the k by score
        idx = idx[row, order]
        all_idx.append(idx)
        all_sim.append(sims[row, idx])
    return np.concatenate(all_idx, 0), np.concatenate(all_sim, 0)


def contexts_for_split(query_df, encoder, db_norm, train_df, k, is_train):
    qvecs = encode_texts(query_df["text"].tolist(), encoder)
    self_idx = np.arange(len(query_df)) if is_train else None   # train rows == first DB rows
    idxs, sims = topk_retrieve(qvecs, db_norm, k, self_global_idx=self_idx)
    contexts = []
    for i in range(len(query_df)):
        neighbours = [
            {"text": train_df.iloc[j]["text"], "sentiment": train_df.iloc[j]["sentiment"]}
            for j in idxs[i]
        ]
        contexts.append(C.build_context_string(neighbours))
    return contexts, idxs, sims


def main():
    print(f"Device: {C.DEVICE}")
    encoder = load_encoder()

    train_df = pd.read_csv(C.DATA_DIR / "train.csv")
    val_df   = pd.read_csv(C.DATA_DIR / "val.csv")
    test_df  = pd.read_csv(C.DATA_DIR / "test.csv")

    # Retrieval database = ALL train embeddings from Task A (the full corpus stays
    # searchable even though we only train the decoder on a subsample).
    db = np.load(C.RES_DIR / "train_embeddings.npy")
    db_norm = db / (np.linalg.norm(db, axis=1, keepdims=True) + 1e-9)
    print(f"Retrieval DB: {db_norm.shape[0]} train embeddings")

    train_sub = train_df.iloc[:TRAIN_N].reset_index(drop=True)
    val_sub   = val_df.iloc[:VAL_N].reset_index(drop=True)
    test_sub  = test_df.iloc[:TEST_N].reset_index(drop=True)

    # ── Precision@k diagnostic on the test subsample ──────────────────────────
    print("\nRetrieval quality (Precision@k = fraction of neighbours sharing the query sentiment):")
    qv = encode_texts(test_sub["text"].tolist(), encoder)
    ks = [1, 3, 5, 10]
    pk_vals = []
    for k in ks:
        idxs, _ = topk_retrieve(qv, db_norm, k)
        hits = [
            np.mean([train_df.iloc[j]["sentiment"] == test_sub.iloc[i]["sentiment"] for j in idxs[i]])
            for i in range(len(test_sub))
        ]
        pk_vals.append(float(np.mean(hits)))
        print(f"  Precision@{k:<2}: {pk_vals[-1]:.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(ks, pk_vals, marker="o")
    plt.xlabel("k"); plt.ylabel("Precision@k (same sentiment)")
    plt.title("Retrieval Precision vs k"); plt.tight_layout()
    plt.savefig(C.RES_DIR / "retrieval_precision.png", dpi=150)
    plt.close()

    # ── A couple of qualitative examples ──────────────────────────────────────
    print("\nExample retrievals:")
    ex_idx, ex_sim = topk_retrieve(qv[:3], db_norm, TOP_K)
    for i in range(3):
        print(f"\n  Query [{test_sub.iloc[i]['sentiment']}]: {test_sub.iloc[i]['text'][:100]}...")
        for j, sc in zip(ex_idx[i], ex_sim[i]):
            ns = train_df.iloc[j]["sentiment"]
            mark = "Y" if ns == test_sub.iloc[i]["sentiment"] else "N"
            print(f"    [{mark}] ({ns}) sim={sc:.4f}: {str(train_df.iloc[j]['text'])[:70]}...")

    # ── Build & save contexts for all three splits ────────────────────────────
    for name, df, is_train in [("train", train_sub, True),
                               ("val",   val_sub,   False),
                               ("test",  test_sub,  False)]:
        ctx, _, _ = contexts_for_split(df, encoder, db_norm, train_df, TOP_K, is_train)
        out = C.RES_DIR / f"{name}_contexts.csv"
        pd.Series(ctx).to_csv(out, index=False, header=["context"])
        print(f"Saved {len(ctx):>5} {name} contexts -> {out.name}")

    print("\nTask B complete.")


if __name__ == "__main__":
    main()
