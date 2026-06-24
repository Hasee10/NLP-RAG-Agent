"""Phase 1 collector — Human Phenotype Ontology (symptoms/phenotypes).

Each HPO term is a clinical phenotype (symptom). We index terms that carry a
definition as `symptom` chunks. Source: hp.obo (open, CC-BY).

Usage:
    python -m medical-rag.collectors.hpo_symptoms
"""

import json
import logging
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from schema import Chunk  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("hpo")

OBO_URL = "https://purl.obolibrary.org/obo/hp.obo"
RAW_DIR = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache" / "hp.obo"

SYSTEM_KW = {
    "cardiovascular": ["cardiac", "heart", "vascular", "arter", "aort"],
    "respiratory": ["pulmonary", "lung", "respiratory", "airway"],
    "neurological": ["neur", "brain", "seizure", "cerebr", "cognit"],
    "gastrointestinal": ["gastrointestinal", "hepatic", "intestin", "stomach", "bowel"],
    "endocrine": ["endocrine", "thyroid", "insulin", "hormone", "glucose"],
    "renal": ["renal", "kidney", "urinary", "bladder"],
    "musculoskeletal": ["skeletal", "bone", "muscle", "joint", "limb"],
    "reproductive": ["genital", "reproductive", "ovar", "testic", "uter"],
    "integumentary": ["skin", "cutaneous", "dermat", "nail", "hair"],
    "immune": ["immun", "autoimmune", "inflammation"],
}


def _body_system(text: str) -> str:
    t = text.lower()
    for system, kws in SYSTEM_KW.items():
        if any(k in t for k in kws):
            return system
    return "general"


def _ensure_obo() -> bool:
    if CACHE.exists():
        return True
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    log.info("downloading hp.obo...")
    try:
        r = requests.get(OBO_URL, timeout=120)
        r.raise_for_status()
        CACHE.write_text(r.text, encoding="utf-8")
        return True
    except Exception as e:
        log.error("download failed: %s", e)
        return False


def collect() -> list[dict]:
    if not _ensure_obo():
        return []
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    chunks: list[dict] = []
    name = definition = None
    in_term = False
    for line in CACHE.read_text(encoding="utf-8").splitlines():
        if line == "[Term]":
            in_term, name, definition = True, None, None
        elif line.startswith("[") and line != "[Term]":
            in_term = False
        elif in_term and line.startswith("name:"):
            name = line[5:].strip()
        elif in_term and line.startswith("def:"):
            # def: "text here" [refs]
            d = line[4:].strip()
            if d.startswith('"'):
                d = d[1:].split('" [')[0].split('"')[0]
            definition = d.strip()
            if name and definition and len(definition) > 20:
                text = f"{name}: {definition}"
                chunks.append(
                    Chunk(text=text, source="hpo", type="symptom",
                          body_system=_body_system(text), tags=[name.lower()]).to_dict()
                )
                name = definition = None
    return chunks


def main():
    chunks = collect()
    out = RAW_DIR / "hpo.json"
    out.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    log.info("wrote %d symptom chunks -> %s", len(chunks), out)


if __name__ == "__main__":
    main()
