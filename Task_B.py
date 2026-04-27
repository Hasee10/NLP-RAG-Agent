import pandas as pd
import numpy as np
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import DataLoader, Dataset
from pathlib import Path

BASE     = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MDL_DIR  = BASE / "models"
RES_DIR  = BASE / "results"
RES_DIR.mkdir(exist_ok=True)

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN   = 128
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM    = 256
NUM_LAYERS = 3
DROPOUT   = 0.1
TOP_K     = 5

with open(DATA_DIR / "vocab.json") as f:
    vocab = json.load(f)

PAD_IDX    = vocab["<PAD>"]
UNK_IDX    = vocab["<UNK>"]
VOCAB_SIZE = len(vocab)


def tokenize_and_encode(text, vocab, max_len):
    tokens = str(text).lower().split()[:max_len]
    ids    = [vocab.get(t, UNK_IDX) for t in tokens]
    ids    = ids + [PAD_IDX] * (max_len - len(ids))
    return ids


def make_pad_mask(seq, pad_idx):
    return seq == pad_idx


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k    = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(attn, v)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.h   = num_heads
        self.d_k = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.attn = ScaledDotProductAttention(dropout)

    def forward(self, x, mask=None):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        out = self.attn(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn  = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff    = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.drop(self.attn(x, mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, embed_dim)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class EncoderModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers,
                 max_len, dropout, num_sent_classes=3, num_len_classes=3):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(embed_dim, max_len, dropout)
        self.layers  = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.sent_head = nn.Linear(embed_dim, num_sent_classes)
        self.len_head  = nn.Linear(embed_dim, num_len_classes)

    def forward(self, x):
        mask = make_pad_mask(x, PAD_IDX).to(x.device)
        out  = self.pos_enc(self.embed(x))
        for layer in self.layers:
            out = layer(out, mask)
        cls_vec = out[:, 0, :]
        return self.sent_head(cls_vec), self.len_head(cls_vec), cls_vec


def encode_text(text, encoder, vocab, max_len):
    ids = tokenize_and_encode(text, vocab, max_len)
    t   = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        _, _, emb = encoder(t)
    v = emb.squeeze(0).cpu().numpy()
    v = v / (np.linalg.norm(v) + 1e-9)
    return v


def retrieve(query_text, encoder, train_embeddings_norm, train_df, k=TOP_K):
    q_vec = encode_text(query_text, encoder, vocab, MAX_LEN).reshape(1, -1)
    sims  = cosine_similarity(q_vec, train_embeddings_norm)[0]
    top_indices = np.argsort(sims)[::-1][:k]
    results = []
    for idx in top_indices:
        results.append({
            "text":         train_df.iloc[idx]["text"],
            "sentiment":    train_df.iloc[idx]["sentiment"],
            "length_label": int(train_df.iloc[idx]["length_label"]),
            "score":        float(sims[idx]),
            "train_idx":    int(idx),
        })
    return results


def build_context_string(retrieved):
    parts = []
    for i, r in enumerate(retrieved):
        parts.append(f"[Example {i+1}] ({r['sentiment']}) {r['text'][:200]}")
    return " | ".join(parts)


def main():
    encoder = EncoderModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
    ).to(DEVICE)
    encoder.load_state_dict(torch.load(MDL_DIR / "encoder_best.pt", map_location=DEVICE))
    encoder.eval()

    train_embeddings = np.load(RES_DIR / "train_embeddings.npy")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df  = pd.read_csv(DATA_DIR / "test.csv")

    norms = np.linalg.norm(train_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    train_embeddings_norm = train_embeddings / norms

    print("Retrieval quality analysis on 10 test samples:")
    print("=" * 80)
    for i in range(10):
        row       = test_df.iloc[i]
        true_sent = row["sentiment"]
        retrieved = retrieve(row["text"], encoder, train_embeddings_norm, train_df, k=TOP_K)
        print(f"\nQuery [{true_sent}]: {row['text'][:120]}...")
        for r in retrieved:
            match = "Y" if r["sentiment"] == true_sent else "N"
            print(f"  [{match}] [{r['sentiment']}] sim={r['score']:.4f}: {r['text'][:80]}...")

    ks    = [1, 3, 5, 10]
    pk_vals = []
    for k in ks:
        hits = []
        for i in range(200):
            row       = test_df.iloc[i]
            retrieved = retrieve(row["text"], encoder, train_embeddings_norm, train_df, k=k)
            h = sum(1 for r in retrieved if r["sentiment"] == row["sentiment"])
            hits.append(h / k)
        pk_vals.append(np.mean(hits))
        print(f"Precision@{k}: {np.mean(hits):.4f}")

    plt.figure(figsize=(6, 4))
    plt.plot(ks, pk_vals, marker="o")
    plt.xlabel("k")
    plt.ylabel("Precision@k (same sentiment)")
    plt.title("Retrieval Precision vs k")
    plt.tight_layout()
    plt.savefig(RES_DIR / "retrieval_precision.png", dpi=150)
    plt.close()

    test_contexts = []
    for i in range(len(test_df)):
        row       = test_df.iloc[i]
        retrieved = retrieve(row["text"], encoder, train_embeddings_norm, train_df, k=TOP_K)
        ctx       = build_context_string(retrieved)
        test_contexts.append(ctx)
        if (i + 1) % 500 == 0:
            print(f"  Retrieved context for {i+1}/{len(test_df)} test samples...")

    pd.Series(test_contexts).to_csv(RES_DIR / "test_contexts.csv", index=False, header=["context"])
    print(f"\nSaved {len(test_contexts)} test contexts -> results/test_contexts.csv")
    print("Retrieval module complete.")


if __name__ == "__main__":
    main()