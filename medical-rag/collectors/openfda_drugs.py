"""Phase 1 collector — OpenFDA drug labels (free REST API, no key required).

Pulls drug label records and extracts indications, dosage, contraindications,
and interactions into canonical Chunk records.

Usage:
    python -m medical-rag.collectors.openfda_drugs --limit 300
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from schema import Chunk  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("openfda")

API = "https://api.fda.gov/drug/label.json"
RAW_DIR = ROOT / "data" / "raw"

# Label sections worth indexing, mapped to a clause that frames the chunk text.
SECTIONS = {
    "indications_and_usage": "Indications and usage",
    "dosage_and_administration": "Dosage and administration",
    "contraindications": "Contraindications",
    "warnings": "Warnings",
    "drug_interactions": "Drug interactions",
    "adverse_reactions": "Adverse reactions",
}


def _names(openfda: dict) -> list[str]:
    out = []
    for k in ("brand_name", "generic_name", "substance_name"):
        out.extend(openfda.get(k, []) or [])
    # dedupe, keep short
    seen, uniq = set(), []
    for n in out:
        nl = n.lower()
        if nl not in seen:
            seen.add(nl)
            uniq.append(n)
    return uniq[:6]


def collect(limit: int, page: int = 100) -> list[dict]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    chunks: list[dict] = []
    fetched = 0
    while fetched < limit:
        n = min(page, limit - fetched)
        try:
            r = requests.get(API, params={"limit": n, "skip": fetched}, timeout=30)
            r.raise_for_status()
            results = r.json().get("results", [])
        except Exception as e:
            log.warning("fetch failed at skip=%d: %s", fetched, e)
            break
        if not results:
            break
        for rec in results:
            ofda = rec.get("openfda", {}) or {}
            names = _names(ofda)
            label = names[0] if names else "This drug"
            tags = [t.lower() for t in names]
            for field_key, clause in SECTIONS.items():
                val = rec.get(field_key)
                if not val:
                    continue
                body = " ".join(val) if isinstance(val, list) else str(val)
                body = body.strip()
                if len(body) < 30:
                    continue
                text = f"{label} — {clause}: {body}"
                try:
                    chunks.append(
                        Chunk(text=text[:4000], source="openfda", type="drug",
                              body_system="general", tags=tags).to_dict()
                    )
                except ValueError as e:
                    log.debug("skip chunk: %s", e)
        fetched += len(results)
        log.info("fetched %d labels, %d chunks so far", fetched, len(chunks))
        time.sleep(0.3)  # be polite to the API
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300, help="number of drug labels to fetch")
    args = ap.parse_args()

    chunks = collect(args.limit)
    out = RAW_DIR / "openfda.json"
    out.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    log.info("wrote %d chunks -> %s", len(chunks), out)


if __name__ == "__main__":
    main()
