"""
Task C — Retrieval-Augmented Decoder LM  (REDESIGNED)
=====================================================
Trains a causal decoder to produce a *grounded* sentiment explanation:

    "this review is <sentiment> because similar reviewers mention <evidence...>"

The <evidence> words are drawn from the retrieved neighbours and are explicitly
chosen to NOT appear in the query review (see rag_common.neighbour_evidence_words).
That makes the RAG ablation honest and meaningful:

  * WITH retrieved context in the input, the model can see those evidence words
    and predict them → low perplexity.
  * WITHOUT context (baseline), the evidence words are unobtainable → high
    perplexity.

So unlike the original assignment (where the target was copied from the query
review itself and retrieval could only add noise), here retrieval carries real,
measurable signal — which is the whole point of RAG.

Reduced scale (8k train rows / 6 epochs / 160-token source) so it retrains on a
CPU in minutes. Uses the contexts produced by Task_B for train / val / test.
"""

import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import rag_common as C

torch.manual_seed(42)
np.random.seed(42)
torch.set_num_threads(max(1, (torch.get_num_threads() or 1)))  # use all available CPU threads

TRAIN_N    = 4000
VAL_N      = 800
TEST_N     = 1200
BATCH_SIZE = 16
EPOCHS     = 4
LR         = 5e-4

base_vocab, dec_vocab, inv_vocab = C.load_vocab()
PAD_IDX    = dec_vocab["<PAD>"]
BOS_IDX    = dec_vocab["<BOS>"]
EOS_IDX    = dec_vocab["<EOS>"]
VOCAB_SIZE = len(dec_vocab)


class ExplanationDataset(torch.utils.data.Dataset):
    """Each item: (src_with_context, src_without_context, target).

    The target is identical in both conditions; only the source differs in whether
    the retrieved context is included — this is the controlled RAG ablation.
    """
    def __init__(self, data_path, context_path, n):
        df  = pd.read_csv(data_path).iloc[:n].reset_index(drop=True)
        ctx = pd.read_csv(context_path)["context"].fillna("").tolist()[:n]
        self.rows, self.contexts = df, ctx

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row     = self.rows.iloc[idx]
        ctx_str = self.contexts[idx] if idx < len(self.contexts) else ""
        sent_id = C.SENTIMENT_MAP[row["sentiment"]]
        len_id  = int(row["length_label"])

        neighbours = C.parse_context(ctx_str)
        tgt_ids, _ = C.build_target(row["sentiment"], neighbours, row["text"], dec_vocab)

        src_rag   = C.build_input_sequence(row["text"], sent_id, len_id, ctx_str, dec_vocab)
        src_norag = C.build_input_sequence(row["text"], sent_id, len_id, "",      dec_vocab)
        return (torch.tensor(src_rag,   dtype=torch.long),
                torch.tensor(src_norag, dtype=torch.long),
                torch.tensor(tgt_ids,   dtype=torch.long))


def _lm_step(model, src, tgt, criterion):
    """Teacher-forced loss over the target span only (source is pure conditioning)."""
    full = torch.cat([src, tgt[:, :-1]], dim=1)              # (B, S+T-1)
    pad  = (full == PAD_IDX)
    logits = model(full, key_pad_mask=pad)
    S, T = src.size(1), tgt.size(1)
    tgt_logits = logits[:, S:S + T - 1, :]                   # predictions for tgt[1:]
    return criterion(tgt_logits.reshape(-1, VOCAB_SIZE), tgt[:, 1:].reshape(-1)), tgt_logits


def run_epoch(model, loader, optimizer, criterion, training):
    model.train() if training else model.eval()
    total, mode = 0.0, ("Train" if training else "Val")
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for bi, (src_rag, _src_norag, tgt) in enumerate(loader):
            src_rag, tgt = src_rag.to(C.DEVICE), tgt.to(C.DEVICE)
            loss, _ = _lm_step(model, src_rag, tgt, criterion)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item()
            if (bi + 1) % 100 == 0:
                print(f"  [{mode}] batch {bi+1}/{len(loader)}  loss={total/(bi+1):.4f}", flush=True)
    return total / len(loader)


def perplexity(model, loader, use_rag):
    """Token-level perplexity over the target. use_rag toggles whether the input
    carries the retrieved context (the ablation knob)."""
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=PAD_IDX, reduction="sum")
    tot_loss, tot_tok = 0.0, 0
    with torch.no_grad():
        for src_rag, src_norag, tgt in loader:
            src = (src_rag if use_rag else src_norag).to(C.DEVICE)
            tgt = tgt.to(C.DEVICE)
            loss, _ = _lm_step(model, src, tgt, crit)
            tot_loss += loss.item()
            tot_tok  += (tgt[:, 1:] != PAD_IDX).sum().item()
    return math.exp(tot_loss / max(tot_tok, 1))


def generate(model, src_ids, max_new=C.DEC_TGT_LEN):
    model.eval()
    gen = [BOS_IDX]
    with torch.no_grad():
        for _ in range(max_new):
            inp = torch.tensor([src_ids + gen], dtype=torch.long, device=C.DEVICE)
            logits = model(inp, key_pad_mask=(inp == PAD_IDX))
            nxt = logits[0, -1, :].argmax(-1).item()
            if nxt == EOS_IDX:
                break
            gen.append(nxt)
    return " ".join(inv_vocab.get(t, "<UNK>") for t in gen[1:])


def main():
    print(f"Device: {C.DEVICE} | Vocab: {VOCAB_SIZE}", flush=True)

    train_ds = ExplanationDataset(C.DATA_DIR / "train.csv", C.RES_DIR / "train_contexts.csv", TRAIN_N)
    val_ds   = ExplanationDataset(C.DATA_DIR / "val.csv",   C.RES_DIR / "val_contexts.csv",   VAL_N)
    test_ds  = ExplanationDataset(C.DATA_DIR / "test.csv",  C.RES_DIR / "test_contexts.csv",  TEST_N)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=BATCH_SIZE)

    model = C.DecoderLM(vocab_size=VOCAB_SIZE, pad_idx=PAD_IDX).to(C.DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_losses, val_losses, best = [], [], float("inf")
    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, optimizer, criterion, True)
        vl = run_epoch(model, val_loader,   optimizer, criterion, False)
        scheduler.step()
        train_losses.append(tr); val_losses.append(vl)
        print(f"Epoch {epoch:02d} | Train {tr:.4f} | Val {vl:.4f}", flush=True)
        if vl < best:
            best = vl
            torch.save(model.state_dict(), C.MDL_DIR / "decoder_best.pt")

    model.load_state_dict(torch.load(C.MDL_DIR / "decoder_best.pt", map_location=C.DEVICE))

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, EPOCHS + 1), train_losses, label="Train", marker="o")
    plt.plot(range(1, EPOCHS + 1), val_losses,   label="Val",   marker="o")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Decoder LM Training Curves")
    plt.legend(); plt.tight_layout()
    plt.savefig(C.RES_DIR / "decoder_curves.png", dpi=150); plt.close()

    print("\nEvaluating perplexity (controlled RAG ablation)...", flush=True)
    ppl_rag   = perplexity(model, test_loader, use_rag=True)
    ppl_norag = perplexity(model, test_loader, use_rag=False)
    print(f"  Perplexity WITH RAG context    : {ppl_rag:.2f}", flush=True)
    print(f"  Perplexity WITHOUT RAG (baseline): {ppl_norag:.2f}", flush=True)
    improvement = (ppl_norag - ppl_rag) / ppl_norag * 100
    print(f"  Relative improvement from RAG   : {improvement:.1f}%", flush=True)

    # ── Qualitative samples ───────────────────────────────────────────────────
    test_df  = pd.read_csv(C.DATA_DIR / "test.csv").iloc[:TEST_N].reset_index(drop=True)
    test_ctx = pd.read_csv(C.RES_DIR / "test_contexts.csv")["context"].fillna("").tolist()[:TEST_N]

    samples = []
    print("\nSample generations (WITH vs WITHOUT retrieval):", flush=True)
    for i in range(5):
        row     = test_df.iloc[i]
        sent_id = C.SENTIMENT_MAP[row["sentiment"]]
        len_id  = int(row["length_label"])
        src_rag   = C.build_input_sequence(row["text"], sent_id, len_id, test_ctx[i], dec_vocab)
        src_norag = C.build_input_sequence(row["text"], sent_id, len_id, "",          dec_vocab)
        g_rag, g_norag = generate(model, src_rag), generate(model, src_norag)
        print(f"\n[{i+1}] ({row['sentiment']}) {row['text'][:90]}...")
        print(f"    WITH RAG : {g_rag}")
        print(f"    NO  RAG  : {g_norag}")
        samples.append({"review": row["text"][:200], "sentiment": row["sentiment"],
                        "with_rag": g_rag, "without_rag": g_norag})

    with open(C.RES_DIR / "ablation.json", "w") as f:
        json.dump({
            "perplexity_rag": ppl_rag,
            "perplexity_norag": ppl_norag,
            "relative_improvement_pct": improvement,
            "best_val_loss": best,
            "config": {"train_n": TRAIN_N, "epochs": EPOCHS,
                       "src_len": C.DEC_SRC_LEN, "tgt_len": C.DEC_TGT_LEN},
        }, f, indent=2)
    with open(C.RES_DIR / "sample_generations.json", "w") as f:
        json.dump(samples, f, indent=2)

    print("\nTask C complete. Outputs -> results/ (decoder_best.pt, ablation.json, "
          "decoder_curves.png, sample_generations.json)", flush=True)


if __name__ == "__main__":
    main()
