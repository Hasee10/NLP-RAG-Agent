import pandas as pd
import numpy as np
import json
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
from pathlib import Path

BASE     = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MDL_DIR  = BASE / "models"
RES_DIR  = BASE / "results"
MDL_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN     = 128
BATCH_SIZE  = 64
EMBED_DIM   = 128
NUM_HEADS   = 4
FF_DIM      = 256
NUM_LAYERS  = 3
DROPOUT     = 0.1
EPOCHS      = 15
LR          = 1e-3
WARMUP_STEPS = 500

torch.manual_seed(42)
np.random.seed(42)

with open(DATA_DIR / "vocab.json") as f:
    vocab = json.load(f)

PAD_IDX    = vocab["<PAD>"]
UNK_IDX    = vocab["<UNK>"]
VOCAB_SIZE = len(vocab)

SENTIMENT_MAP = {"Negative": 0, "Neutral": 1, "Positive": 2}


def tokenize_and_encode(text, vocab, max_len):
    tokens = str(text).lower().split()[:max_len]
    ids    = [vocab.get(t, UNK_IDX) for t in tokens]
    ids    = ids + [PAD_IDX] * (max_len - len(ids))
    return ids


class ReviewDataset(Dataset):
    def __init__(self, path, vocab, max_len):
        self.df     = pd.read_csv(path)
        self.vocab  = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row    = self.df.iloc[idx]
        ids    = tokenize_and_encode(str(row["text"]), self.vocab, self.max_len)
        sent   = SENTIMENT_MAP[row["sentiment"]]
        length = int(row["length_label"])
        return (
            torch.tensor(ids,    dtype=torch.long),
            torch.tensor(sent,   dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
        )


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
        assert embed_dim % num_heads == 0
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
                 max_len, dropout, num_sent_classes, num_len_classes):
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


def get_lr(step, embed_dim, warmup):
    if step == 0:
        step = 1
    return embed_dim ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))


def run_epoch(model, loader, optimizer, scheduler, criterion_s, criterion_l, training):
    model.train() if training else model.eval()
    total_loss = 0
    all_s_pred, all_s_true = [], []
    all_l_pred, all_l_true = [], []

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for ids, sent, length in loader:
            ids, sent, length = ids.to(DEVICE), sent.to(DEVICE), length.to(DEVICE)
            s_logits, l_logits, _ = model(ids)
            loss_s = criterion_s(s_logits, sent)
            loss_l = criterion_l(l_logits, length)
            loss   = loss_s + 0.5 * loss_l

            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            total_loss += loss.item()
            all_s_pred.extend(s_logits.argmax(-1).cpu().tolist())
            all_s_true.extend(sent.cpu().tolist())
            all_l_pred.extend(l_logits.argmax(-1).cpu().tolist())
            all_l_true.extend(length.cpu().tolist())

    avg_loss = total_loss / len(loader)
    f1_s = f1_score(all_s_true, all_s_pred, average="macro", zero_division=0)
    f1_l = f1_score(all_l_true, all_l_pred, average="macro", zero_division=0)
    return avg_loss, f1_s, f1_l


def main():
    train_ds = ReviewDataset(DATA_DIR / "train.csv", vocab, MAX_LEN)
    val_ds   = ReviewDataset(DATA_DIR / "val.csv",   vocab, MAX_LEN)
    test_ds  = ReviewDataset(DATA_DIR / "test.csv",  vocab, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    model = EncoderModel(
        vocab_size=VOCAB_SIZE,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN,
        dropout=DROPOUT,
        num_sent_classes=3,
        num_len_classes=3,
    ).to(DEVICE)

    train_df_full = pd.read_csv(DATA_DIR / "train.csv")
    sent_counts   = train_df_full["sentiment"].map(SENTIMENT_MAP).value_counts().sort_index()
    sent_weights  = (1.0 / sent_counts.values).tolist()
    sent_weights  = torch.tensor([w / sum(sent_weights) for w in sent_weights], dtype=torch.float).to(DEVICE)

    criterion_s = nn.CrossEntropyLoss(weight=sent_weights)
    criterion_l = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: get_lr(step, EMBED_DIM, WARMUP_STEPS)
    )

    history = {
        "train_loss": [], "val_loss": [],
        "train_f1_sent": [], "val_f1_sent": [],
        "train_f1_len":  [], "val_f1_len":  [],
    }

    best_val_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_f1s, tr_f1l = run_epoch(model, train_loader, optimizer, scheduler, criterion_s, criterion_l, True)
        vl_loss, vl_f1s, vl_f1l = run_epoch(model, val_loader,   optimizer, scheduler, criterion_s, criterion_l, False)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_f1_sent"].append(tr_f1s)
        history["val_f1_sent"].append(vl_f1s)
        history["train_f1_len"].append(tr_f1l)
        history["val_f1_len"].append(vl_f1l)

        print(f"Epoch {epoch:02d} | Train Loss {tr_loss:.4f} | Val Loss {vl_loss:.4f} | Val F1-Sent {vl_f1s:.4f} | Val F1-Len {vl_f1l:.4f}")

        if vl_f1s > best_val_f1:
            best_val_f1 = vl_f1s
            torch.save(model.state_dict(), MDL_DIR / "encoder_best.pt")

    model.load_state_dict(torch.load(MDL_DIR / "encoder_best.pt", map_location=DEVICE))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history["train_loss"],    label="Train"); axes[0].plot(history["val_loss"],     label="Val"); axes[0].set_title("Loss");            axes[0].legend()
    axes[1].plot(history["train_f1_sent"], label="Train"); axes[1].plot(history["val_f1_sent"],  label="Val"); axes[1].set_title("F1 — Sentiment");  axes[1].legend()
    axes[2].plot(history["train_f1_len"],  label="Train"); axes[2].plot(history["val_f1_len"],   label="Val"); axes[2].set_title("F1 — Length Label"); axes[2].legend()
    plt.tight_layout()
    plt.savefig(RES_DIR / "encoder_curves.png", dpi=150)
    plt.close()

    model.eval()
    all_s_pred, all_s_true = [], []
    all_l_pred, all_l_true = [], []
    with torch.no_grad():
        for ids, sent, length in test_loader:
            ids = ids.to(DEVICE)
            s_logits, l_logits, _ = model(ids)
            all_s_pred.extend(s_logits.argmax(-1).cpu().tolist())
            all_s_true.extend(sent.tolist())
            all_l_pred.extend(l_logits.argmax(-1).cpu().tolist())
            all_l_true.extend(length.tolist())

    print("\nSentiment Classification Report (Test):")
    print(classification_report(all_s_true, all_s_pred, target_names=["Negative", "Neutral", "Positive"]))
    print("Length Label Classification Report (Test):")
    print(classification_report(all_l_true, all_l_pred, target_names=["Short", "Medium", "Long"]))

    full_train_loader = DataLoader(train_ds, batch_size=256, shuffle=False, num_workers=0, pin_memory=False)
    all_embeddings = []
    with torch.no_grad():
        for ids, _, _ in full_train_loader:
            ids = ids.to(DEVICE)
            _, _, emb = model(ids)
            all_embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(all_embeddings, axis=0)
    np.save(RES_DIR / "train_embeddings.npy", embeddings)
    print(f"\nSaved {embeddings.shape[0]} training embeddings -> results/train_embeddings.npy")


if __name__ == "__main__":
    main()