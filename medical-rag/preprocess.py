"""Phase 2 — preprocessing.

raw/*.json  ->  clean -> chunk (tiktoken) -> dedup (MinHash LSH) -> corpus.jsonl
            ->  MedQuAD hard-negative mining (BM25) -> train/val.jsonl (stratified)

Usage:
    python -m medical-rag.preprocess --max-anchors 6000
"""

import argparse
import glob
import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import tiktoken
from datasketch import MinHash, MinHashLSH
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))
from schema import Chunk  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("preprocess")

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
ENC = tiktoken.get_encoding("cl100k_base")

MAX_TOK, OVERLAP = 280, 50
_HTML = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    return _WS.sub(" ", _HTML.sub(" ", str(text))).strip()


def chunk_text(text: str) -> list[str]:
    toks = ENC.encode(text)
    if len(toks) <= MAX_TOK:
        return [text]
    out, i = [], 0
    while i < len(toks):
        out.append(ENC.decode(toks[i:i + MAX_TOK]))
        i += MAX_TOK - OVERLAP
    return out


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def load_raw() -> list[dict]:
    recs = []
    for f in glob.glob(str(RAW / "*.json")):
        recs.extend(json.load(open(f, encoding="utf-8")))
    return recs


def build_corpus(raw: list[dict]) -> list[dict]:
    """Clean + chunk every record, then drop near-duplicates with MinHash LSH."""
    lsh = MinHashLSH(threshold=0.85, num_perm=64)
    corpus, kept = [], 0
    for r in raw:
        base_text = clean(r.get("text") or r.get("answer") or "")
        if len(base_text) < 20:
            continue
        for j, piece in enumerate(chunk_text(base_text)):
            mh = MinHash(num_perm=64)
            for sh in set(_tokens(piece)):
                mh.update(sh.encode("utf-8"))
            key = f"{r['id']}_{j}"
            if lsh.query(mh):           # near-duplicate of something already kept
                continue
            lsh.insert(key, mh)
            corpus.append({
                "id": key,
                "text": piece,
                "source": r["source"],
                "type": r["type"],
                "body_system": r.get("body_system", "general"),
                "severity_level": r.get("severity_level"),
                "tags": r.get("tags", []),
            })
            kept += 1
    log.info("corpus: %d chunks after chunk+dedup (from %d raw records)", kept, len(raw))
    return corpus


def mine_pairs(raw: list[dict], corpus: list[dict], max_anchors: int) -> list[dict]:
    """For each MedQuAD (question, answer), mine 3 BM25 hard negatives from the corpus."""
    corpus_tokens = [_tokens(c["text"]) for c in corpus]
    bm25 = BM25Okapi(corpus_tokens)
    anchors = [r for r in raw if r.get("type") == "qa_pair" and r.get("question")]
    if max_anchors:
        anchors = anchors[:max_anchors]
    log.info("mining hard negatives for %d anchors over %d corpus docs", len(anchors), len(corpus))

    pairs = []
    for n, a in enumerate(anchors):
        q, pos = a["question"], clean(a["answer"])
        scores = bm25.get_scores(_tokens(q))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:12]
        negs = []
        pos_head = pos[:80].lower()
        for i in ranked:
            ctext = corpus[i]["text"]
            if ctext[:80].lower() == pos_head:   # skip the true positive
                continue
            negs.append(ctext[:1000])
            if len(negs) == 3:
                break
        if len(negs) < 3:
            continue
        pairs.append({
            "anchor": q,
            "positive": pos[:1000],
            "negatives": negs,
            "body_system": a.get("body_system", "general"),
        })
        if (n + 1) % 1000 == 0:
            log.info("  mined %d/%d", n + 1, len(anchors))
    return pairs


def stratified_split(pairs: list[dict], val_frac: float = 0.15):
    by_sys = defaultdict(list)
    for p in pairs:
        by_sys[p["body_system"]].append(p)
    train, val = [], []
    for _, group in by_sys.items():
        cut = max(1, int(len(group) * val_frac))
        val.extend(group[:cut])
        train.extend(group[cut:])
    return train, val


def write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-anchors", type=int, default=6000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    log.info("loaded %d raw records", len(raw))

    corpus = build_corpus(raw)
    write_jsonl(OUT / "corpus.jsonl", corpus)

    pairs = mine_pairs(raw, corpus, args.max_anchors)
    train, val = stratified_split(pairs)
    write_jsonl(OUT / "train.jsonl", train)
    write_jsonl(OUT / "val.jsonl", val)

    log.info("DONE  corpus=%d  train=%d  val=%d", len(corpus), len(train), len(val))


if __name__ == "__main__":
    main()
