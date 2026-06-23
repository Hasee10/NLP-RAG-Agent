"""
rag_common.py
=============
Single source of truth for the RAG agent: model architectures, tokenizer,
vocabulary handling, the I/O sequence format, and the *neighbour-grounded
target* used to train and evaluate the decoder.

Both the training scripts (Task_B / Task_C) and the deployable inference agent
(rag_agent.py) import from this file, which guarantees the decoder checkpoint is
always loaded with the exact architecture and sequence layout it was trained on.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Paths & device
# ──────────────────────────────────────────────────────────────────────────────
BASE     = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MDL_DIR  = BASE / "models"
RES_DIR  = BASE / "results"
MDL_DIR.mkdir(exist_ok=True)
RES_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ──────────────────────────────────────────────────────────────────────────────
# Architecture config (shared, frozen — must match the saved checkpoints)
# ──────────────────────────────────────────────────────────────────────────────
EMBED_DIM  = 128
NUM_HEADS  = 4
FF_DIM     = 256
NUM_LAYERS = 3
DROPOUT    = 0.1

# Encoder (Task A) — these MUST match Task_A.py so encoder_best.pt loads cleanly.
ENC_MAX_LEN = 128

# Decoder (Task C) — reduced from the original 256/40 to keep CPU retraining fast.
# The decoder's positional-encoding buffer has length DEC_SRC_LEN + DEC_TGT_LEN,
# so these values are baked into decoder_best.pt and must not change after training.
DEC_SRC_LEN = 96           # [BOS][SENT][LEN] + review + | + retrieved context
DEC_TGT_LEN = 28           # [BOS] explanation ... [EOS]
DEC_MAX_LEN = DEC_SRC_LEN + DEC_TGT_LEN
DEC_REVIEW_TOK  = 48       # review tokens kept in the source
DEC_CONTEXT_TOK = 40       # retrieved-context tokens kept in the source

SENTIMENT_MAP   = {"Negative": 0, "Neutral": 1, "Positive": 2}
SENTIMENT_WORDS = {"Negative": "negative", "Neutral": "neutral", "Positive": "positive"}
SENT_TOKENS = ["[NEG]", "[NEU]", "[POS]"]
LEN_TOKENS  = ["[SHORT]", "[MED]", "[LONG]"]


# ──────────────────────────────────────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────────────────────────────────────
def load_vocab():
    """Return (base_vocab, decoder_vocab, inv_decoder_vocab).

    base_vocab    : the vocab written by clean_data.py — used by the ENCODER.
    decoder_vocab : base vocab + sentiment/length control tokens — used by the DECODER.
    """
    with open(DATA_DIR / "vocab.json") as f:
        base_vocab = json.load(f)

    dec_vocab = dict(base_vocab)
    for tok in SENT_TOKENS + LEN_TOKENS:
        if tok not in dec_vocab:
            dec_vocab[tok] = len(dec_vocab)
    inv = {v: k for k, v in dec_vocab.items()}
    return base_vocab, dec_vocab, inv


def tokenize(text, vocab, max_len):
    unk = vocab["<UNK>"]
    tokens = str(text).lower().split()[:max_len]
    return [vocab.get(t, unk) for t in tokens]


def words_of(text):
    """Lower-cased alphabetic word list, used for evidence extraction."""
    return re.findall(r"[a-z']+", str(text).lower())


# ──────────────────────────────────────────────────────────────────────────────
# Transformer building blocks (shared by encoder & decoder)
# ──────────────────────────────────────────────────────────────────────────────
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

    def forward(self, x, attn_mask=None, key_pad_mask=None):
        B, T, _ = x.shape
        q = self.W_q(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        out = self.attn(q, k, v, attn_mask=attn_mask, key_pad_mask=key_pad_mask)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)


class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout, activation="relu"):
        super().__init__()
        act = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


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


# ──────────────────────────────────────────────────────────────────────────────
# Encoder (Task A) — architecture identical to Task_A.py so encoder_best.pt loads
# ──────────────────────────────────────────────────────────────────────────────
class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.attn  = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff    = FeedForward(embed_dim, ff_dim, dropout, activation="relu")
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, key_pad_mask=None):
        x = self.norm1(x + self.drop(self.attn(x, key_pad_mask=key_pad_mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class EncoderModel(nn.Module):
    def __init__(self, vocab_size, pad_idx,
                 embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM,
                 num_layers=NUM_LAYERS, max_len=ENC_MAX_LEN, dropout=DROPOUT,
                 num_sent_classes=3, num_len_classes=3):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(embed_dim, max_len, dropout)
        self.layers  = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.sent_head = nn.Linear(embed_dim, num_sent_classes)
        self.len_head  = nn.Linear(embed_dim, num_len_classes)

    def forward(self, x):
        mask = (x == self.pad_idx).to(x.device)
        out  = self.pos_enc(self.embed(x))
        for layer in self.layers:
            out = layer(out, key_pad_mask=mask)
        cls_vec = out[:, 0, :]
        return self.sent_head(cls_vec), self.len_head(cls_vec), cls_vec


# ──────────────────────────────────────────────────────────────────────────────
# Decoder (Task C) — causal LM
# ──────────────────────────────────────────────────────────────────────────────
class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff        = FeedForward(embed_dim, ff_dim, dropout, activation="gelu")
        self.norm1     = nn.LayerNorm(embed_dim)
        self.norm2     = nn.LayerNorm(embed_dim)
        self.drop      = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None, key_pad_mask=None):
        x = self.norm1(x + self.drop(self.self_attn(x, attn_mask=causal_mask, key_pad_mask=key_pad_mask)))
        x = self.norm2(x + self.drop(self.ff(x)))
        return x


class DecoderLM(nn.Module):
    def __init__(self, vocab_size, pad_idx,
                 embed_dim=EMBED_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM,
                 num_layers=NUM_LAYERS, max_len=DEC_MAX_LEN, dropout=DROPOUT):
        super().__init__()
        self.pad_idx = pad_idx
        self.embed   = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(embed_dim, max_len, dropout)
        self.layers  = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.norm    = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def _causal_mask(self, T, device):
        mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).bool()
        return torch.zeros(T, T, device=device).masked_fill(mask, float("-inf"))

    def forward(self, x, key_pad_mask=None):
        T      = x.size(1)
        causal = self._causal_mask(T, x.device)
        out    = self.pos_enc(self.embed(x))
        for layer in self.layers:
            out = layer(out, causal_mask=causal, key_pad_mask=key_pad_mask)
        return self.lm_head(self.norm(out))


# ──────────────────────────────────────────────────────────────────────────────
# Sequence construction (input) and the neighbour-grounded target
# ──────────────────────────────────────────────────────────────────────────────
def build_input_sequence(review_text, sentiment_id, length_id, context_str, dec_vocab):
    """[BOS][SENT][LEN] review  |  retrieved-context  → padded to DEC_SRC_LEN."""
    pad = dec_vocab["<PAD>"]
    bos = dec_vocab["<BOS>"]
    unk = dec_vocab["<UNK>"]
    sep = dec_vocab.get("|", unk)
    sent_tok = dec_vocab.get(SENT_TOKENS[sentiment_id], unk)
    len_tok  = dec_vocab.get(LEN_TOKENS[length_id], unk)

    review_ids  = tokenize(review_text, dec_vocab, DEC_REVIEW_TOK)
    context_ids = tokenize(context_str, dec_vocab, DEC_CONTEXT_TOK)
    seq = [bos, sent_tok, len_tok] + review_ids + [sep] + context_ids
    seq = seq[:DEC_SRC_LEN]
    seq = seq + [pad] * (DEC_SRC_LEN - len(seq))
    return seq


# A compact stop-list so evidence words are content-bearing, not glue words.
_STOP = set("""
the a an and or but if then else of to in on for with from by as at is are was were be been being
this that these those it its they them their there here he she his her you your i we our us my me
have has had do does did will would can could should may might must not no nor so very too just
about into over under out up down off again more most some such only own same than that's i've
it's they're product item buy bought use used using one two get got would also really much many
""".split())


def neighbour_evidence_words(neighbour_texts, query_text, k=3):
    """Top-k content words that appear across the retrieved neighbours but NOT in
    the query review. These are, by construction, only obtainable via retrieval —
    which is precisely what lets the RAG ablation be honest and meaningful."""
    qwords = set(words_of(query_text))
    cnt = Counter()
    for nt in neighbour_texts:
        for w in set(words_of(nt)):           # set() → count documents, not raw freq
            if len(w) >= 4 and w not in _STOP and w not in qwords and w.isalpha():
                cnt[w] += 1
    return [w for w, _ in cnt.most_common(k)]


def build_target(sentiment_label, neighbour_texts, query_text, dec_vocab):
    """Reference explanation whose evidence is grounded in the retrieved neighbours.

    Returns (token_ids, text). The content words after 'mention' come only from the
    neighbours, so a model WITHOUT the retrieved context cannot predict them — the
    perplexity gap between the RAG and no-RAG conditions reflects real retrieval value.
    """
    pad = dec_vocab["<PAD>"]
    bos = dec_vocab["<BOS>"]
    eos = dec_vocab["<EOS>"]
    label_word = SENTIMENT_WORDS[sentiment_label]
    evidence   = neighbour_evidence_words(neighbour_texts, query_text, 3)

    if evidence:
        text = f"this review is {label_word} because similar reviewers mention {' '.join(evidence)}"
    else:
        text = f"this review is {label_word} overall"

    ids = [bos] + tokenize(text, dec_vocab, DEC_TGT_LEN - 2) + [eos]
    ids = ids[:DEC_TGT_LEN]
    ids = ids + [pad] * (DEC_TGT_LEN - len(ids))
    return ids, text


def parse_context(context_str):
    """Split a saved context string back into the list of neighbour review texts.
    Format produced by Task_B: '[Example 1] (Sentiment) text | [Example 2] ...'."""
    if not isinstance(context_str, str) or not context_str.strip():
        return []
    parts = context_str.split(" | ")
    out = []
    for p in parts:
        p = re.sub(r"^\[Example \d+\]\s*\([^)]*\)\s*", "", p).strip()
        if p:
            out.append(p)
    return out


def build_context_string(neighbours):
    """neighbours: list of dicts with 'sentiment' and 'text'. Inverse of parse_context."""
    return " | ".join(
        f"[Example {i+1}] ({n['sentiment']}) {str(n['text'])[:200]}"
        for i, n in enumerate(neighbours)
    )
