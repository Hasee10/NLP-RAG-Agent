import sys
import pandas as pd
import numpy as np
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

torch.manual_seed(42)
np.random.seed(42)

BASE     = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MDL_DIR  = BASE / "models"
RES_DIR  = BASE / "results"
MDL_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SRC_LEN = 256
MAX_TGT_LEN = 40
BATCH_SIZE  = 8
EMBED_DIM   = 128
NUM_HEADS   = 4
FF_DIM      = 256
NUM_LAYERS  = 3
DROPOUT     = 0.1
EPOCHS      = 12
LR          = 5e-4

with open(DATA_DIR / "vocab.json") as f:
    vocab = json.load(f)

PAD_IDX = vocab["<PAD>"]
UNK_IDX = vocab["<UNK>"]
BOS_IDX = vocab["<BOS>"]
EOS_IDX = vocab["<EOS>"]

SENT_TOKENS = ["[NEG]", "[NEU]", "[POS]"]
LEN_TOKENS  = ["[SHORT]", "[MED]", "[LONG]"]
for tok in SENT_TOKENS + LEN_TOKENS:
    if tok not in vocab:
        vocab[tok] = len(vocab)

VOCAB_SIZE = len(vocab)
inv_vocab  = {v: k for k, v in vocab.items()}

SENTIMENT_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}


def tokenize(text, max_len):
    tokens = str(text).lower().split()[:max_len]
    return [vocab.get(t, UNK_IDX) for t in tokens]


def build_input_sequence(review_text, sentiment, length_label, context_str, max_len):
    sent_tok_id = vocab.get(SENT_TOKENS[sentiment], UNK_IDX)
    len_tok_id  = vocab.get(LEN_TOKENS[length_label], UNK_IDX)
    review_ids  = tokenize(review_text, 100)
    context_ids = tokenize(context_str, 120)
    seq = [BOS_IDX, sent_tok_id, len_tok_id] + review_ids + [vocab.get("|", UNK_IDX)] + context_ids
    seq = seq[:max_len]
    seq = seq + [PAD_IDX] * (max_len - len(seq))
    return seq


def build_reference(sentiment_label, text):
    label_word = {"Negative": "negative", "Neutral": "neutral", "Positive": "positive"}[sentiment_label]
    excerpt    = " ".join(str(text).split()[:12])
    ref        = f"this review is {label_word} because {excerpt}"
    ids        = [BOS_IDX] + tokenize(ref, MAX_TGT_LEN - 2) + [EOS_IDX]
    ids        = ids + [PAD_IDX] * (MAX_TGT_LEN - len(ids))
    return ids[:MAX_TGT_LEN]


class ExplanationDataset(Dataset):
    def __init__(self, data_path, context_path):
        self.df       = pd.read_csv(data_path)
        ctx_df        = pd.read_csv(context_path)
        self.contexts = ctx_df["context"].fillna("").tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        ctx     = self.contexts[idx] if idx < len(self.contexts) else ""
        sent_id = SENTIMENT_MAP[row["sentiment"]]
        len_id  = int(row["length_label"])
        src = build_input_sequence(row["text"], sent_id, len_id, ctx, MAX_SRC_LEN)
        tgt = build_reference(row["sentiment"], row["text"])
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, attn_mask=None, key_pad_mask=None):
        d_k    = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if attn_mask is not None:
            scores = scores + attn_mask
        if key_pad_mask is not None:
            scores = scores.masked_fill(key_pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = self.dropout(F.softmax(scores, dim=-1))
        return torch.matmul(attn, v)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super().__init__()
        self.h   = num_heads
        self.d_k = embed_dim // num_heads
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)
        self.attn = ScaledDotProductAttention(dropout)

    def forward(self, x, causal_mask=None, key_pad_mask=None):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        out = self.attn(q, k, v, attn_mask=causal_mask, key_pad_mask=key_pad_mask)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ff        = FeedForward(embed_dim, ff_dim, dropout)
        self.norm1     = nn.LayerNorm(embed_dim)
        self.norm2     = nn.LayerNorm(embed_dim)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None, key_pad_mask=None):
        x = self.norm1(x + self.drop(self.self_attn(x, causal_mask=causal_mask, key_pad_mask=key_pad_mask)))
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


class DecoderLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, num_layers, max_len, dropout):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)
        self.pos_enc = PositionalEncoding(embed_dim, max_len, dropout)
        self.layers  = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm    = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def make_causal_mask(self, T):
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
        return mask.float().masked_fill(mask, float("-inf")).to(DEVICE)

    def forward(self, x, key_pad_mask=None):
        T      = x.size(1)
        causal = self.make_causal_mask(T)
        out    = self.pos_enc(self.embed(x))
        for layer in self.layers:
            out = layer(out, causal_mask=causal, key_pad_mask=key_pad_mask)
        return self.lm_head(self.norm(out))


# ── FIX: single correct run_epoch ─────────────────────────────────────────────
# full_seq = [src | tgt[:,:-1]]  shape: (B, S+T-1)
# tgt[:,1:] shape: (B, T-1)
# Slice must give exactly T-1 positions → start at S, not S-1
def run_epoch(model, loader, optimizer, criterion, training):
    model.train() if training else model.eval()
    total_loss = 0
    mode = "Train" if training else "Val"
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch_idx, (src, tgt) in enumerate(loader):
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            full_seq = torch.cat([src, tgt[:, :-1]], dim=1)        # (B, S+T-1)
            pad_mask = (full_seq == PAD_IDX)
            logits   = model(full_seq, key_pad_mask=pad_mask)       # (B, S+T-1, V)
            S = src.size(1)
            T = tgt.size(1)
            tgt_logits = logits[:, S : S + T - 1, :]               # (B, T-1, V)
            loss = criterion(
                tgt_logits.reshape(-1, VOCAB_SIZE),
                tgt[:, 1:].reshape(-1)                              # (B*(T-1),)
            )
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                print(f"  [{mode}] batch {batch_idx+1}/{len(loader)}  loss={total_loss/(batch_idx+1):.4f}", flush=True)
    return total_loss / len(loader)


# ── FIX: single correct compute_perplexity ────────────────────────────────────
def compute_perplexity(model, loader):
    model.eval()
    total_loss   = 0
    total_tokens = 0
    criterion    = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction="sum")
    with torch.no_grad():
        for src, tgt in loader:
            src, tgt = src.to(DEVICE), tgt.to(DEVICE)
            full_seq = torch.cat([src, tgt[:, :-1]], dim=1)
            pad_mask = (full_seq == PAD_IDX)
            logits   = model(full_seq, key_pad_mask=pad_mask)
            S = src.size(1)
            T = tgt.size(1)
            tgt_logits = logits[:, S : S + T - 1, :]               # (B, T-1, V)
            loss     = criterion(tgt_logits.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1))
            non_pad  = (tgt[:, 1:] != PAD_IDX).sum().item()
            total_loss   += loss.item()
            total_tokens += non_pad
    return math.exp(total_loss / max(total_tokens, 1))


def generate(model, src_ids, max_new=MAX_TGT_LEN):
    model.eval()
    generated = [BOS_IDX]
    with torch.no_grad():
        for _ in range(max_new):
            inp      = torch.tensor([src_ids + generated], dtype=torch.long).to(DEVICE)
            pad_mask = (inp == PAD_IDX)
            logits   = model(inp, key_pad_mask=pad_mask)
            next_tok = logits[0, -1, :].argmax(-1).item()
            if next_tok == EOS_IDX:
                break
            generated.append(next_tok)
    return " ".join(inv_vocab.get(t, "<UNK>") for t in generated[1:])


def make_dummy_ctx_file(n, path):
    pd.Series([""] * n).to_csv(path, index=False, header=["context"])


def main():
    print("Starting Task C: Decoder LM Training", flush=True)
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df   = pd.read_csv(DATA_DIR / "val.csv")
    test_df  = pd.read_csv(DATA_DIR / "test.csv")

    test_ctx_path = RES_DIR / "test_contexts.csv"
    dummy_val     = RES_DIR / "dummy_val_ctx.csv"
    dummy_test_nr = RES_DIR / "dummy_test_norag_ctx.csv"
    dummy_train   = RES_DIR / "dummy_train_ctx.csv"

    make_dummy_ctx_file(len(train_df), dummy_train)
    make_dummy_ctx_file(len(val_df),   dummy_val)
    make_dummy_ctx_file(len(test_df),  dummy_test_nr)

    train_ds_rag  = ExplanationDataset(DATA_DIR / "train.csv", dummy_train)
    val_ds_rag    = ExplanationDataset(DATA_DIR / "val.csv",   dummy_val)
    test_ds_rag   = ExplanationDataset(DATA_DIR / "test.csv",  test_ctx_path)
    test_ds_norag = ExplanationDataset(DATA_DIR / "test.csv",  dummy_test_nr)

    train_loader      = DataLoader(train_ds_rag,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader        = DataLoader(val_ds_rag,    batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader_rag   = DataLoader(test_ds_rag,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader_norag = DataLoader(test_ds_norag, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    model = DecoderLM(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        max_len=MAX_SRC_LEN + MAX_TGT_LEN,
        dropout=DROPOUT,
    ).to(DEVICE)
    print(f"Model created on device: {DEVICE}", flush=True)
    print(f"Vocabulary size: {VOCAB_SIZE}", flush=True)

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_losses, val_losses = [], []
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, optimizer, criterion, True)
        vl = run_epoch(model, val_loader,   optimizer, criterion, False)
        scheduler.step()
        train_losses.append(tr)
        val_losses.append(vl)
        print(f"Epoch {epoch:02d} | Train Loss {tr:.4f} | Val Loss {vl:.4f}", flush=True)
        if vl < best_val_loss:
            best_val_loss = vl
            torch.save(model.state_dict(), MDL_DIR / "decoder_best.pt")

    model.load_state_dict(torch.load(MDL_DIR / "decoder_best.pt", map_location=DEVICE))

    # ── Training curves ───────────────────────────────────────────────────────
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Decoder LM Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RES_DIR / "decoder_curves.png", dpi=150)
    plt.close()
    print("Saved training curves -> results/decoder_curves.png", flush=True)

    # ── Perplexity evaluation ─────────────────────────────────────────────────
    print("\nEvaluating perplexity...", flush=True)
    ppl_rag   = compute_perplexity(model, test_loader_rag)
    ppl_norag = compute_perplexity(model, test_loader_norag)
    print(f"\nPerplexity (with RAG context)    : {ppl_rag:.2f}", flush=True)
    print(f"Perplexity (without RAG context) : {ppl_norag:.2f}", flush=True)

    # ── Generated explanations ────────────────────────────────────────────────
    test_ctx_list = pd.read_csv(test_ctx_path)["context"].fillna("").tolist()

    print("\n" + "=" * 80, flush=True)
    print("Generated Explanations — WITH RAG", flush=True)
    print("=" * 80, flush=True)
    for i in range(5):
        row     = test_df.iloc[i]
        sent_id = SENTIMENT_MAP[row["sentiment"]]
        len_id  = int(row["length_label"])
        ctx     = test_ctx_list[i] if i < len(test_ctx_list) else ""
        src_ids = build_input_sequence(row["text"], sent_id, len_id, ctx, MAX_SRC_LEN)
        expl    = generate(model, src_ids)
        print(f"\n[{i+1}] Review      : {row['text'][:120]}...", flush=True)
        print(f"    True Sent   : {row['sentiment']}", flush=True)
        print(f"    Explanation : {expl}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("Generated Explanations — WITHOUT RAG (Baseline)", flush=True)
    print("=" * 80, flush=True)
    for i in range(5):
        row     = test_df.iloc[i]
        sent_id = SENTIMENT_MAP[row["sentiment"]]
        len_id  = int(row["length_label"])
        src_ids = build_input_sequence(row["text"], sent_id, len_id, "", MAX_SRC_LEN)
        expl    = generate(model, src_ids)
        print(f"\n[{i+1}] Review      : {row['text'][:120]}...", flush=True)
        print(f"    True Sent   : {row['sentiment']}", flush=True)
        print(f"    Explanation : {expl}", flush=True)

    # ── RAG ablation summary ──────────────────────────────────────────────────
    print(f"\nRAG Ablation Summary:", flush=True)
    print(f"  With retrieval    -> Perplexity: {ppl_rag:.2f}", flush=True)
    print(f"  Without retrieval -> Perplexity: {ppl_norag:.2f}", flush=True)
    print(f"  Reduction         : {ppl_norag - ppl_rag:.2f} points", flush=True)

    import json as _json
    with open(RES_DIR / "ablation.json", "w") as f:
        _json.dump({"perplexity_rag": ppl_rag, "perplexity_norag": ppl_norag}, f, indent=2)

    print("\nDecoder complete. All outputs saved to results/", flush=True)


if __name__ == "__main__":
    main()