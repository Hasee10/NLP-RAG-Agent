"""Phase 1 collector — MedQuAD medical QA pairs (NIH/CDC/NLM, via GitHub).

These (question, answer) pairs are the anchors/positives for Phase 3 contrastive
training, and their answers become retrievable corpus chunks. Some answers were
removed from the public repo for licensing — those pairs are skipped.

Usage:
    python -m medical-rag.collectors.medquad_qa
"""

import json
import logging
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from schema import Chunk  # noqa: E402  (validates body_system/type; reused for tagging)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("medquad")

REPO = "https://github.com/abachaa/MedQuAD.git"
CACHE = ROOT / "data" / "cache" / "MedQuAD"
RAW_DIR = ROOT / "data" / "raw"

# Light keyword → body_system mapping (best-effort; defaults to general).
SYSTEM_KW = {
    "cardiovascular": ["heart", "cardiac", "artery", "blood pressure", "coronary", "hypertension"],
    "respiratory": ["lung", "asthma", "respiratory", "pneumonia", "copd", "breath"],
    "neurological": ["brain", "nerve", "seizure", "stroke", "alzheimer", "parkinson", "migraine"],
    "gastrointestinal": ["stomach", "liver", "intestin", "bowel", "colon", "digest", "hepat"],
    "endocrine": ["diabetes", "thyroid", "insulin", "hormone", "adrenal"],
    "renal": ["kidney", "renal", "bladder", "urinary"],
    "musculoskeletal": ["bone", "joint", "muscle", "arthritis", "spine", "fracture"],
    "reproductive": ["pregnan", "ovar", "uterus", "prostate", "menstru"],
    "integumentary": ["skin", "rash", "dermat", "eczema"],
    "immune": ["immune", "hiv", "lupus", "allergy", "autoimmune"],
}


def _body_system(text: str) -> str:
    t = text.lower()
    for system, kws in SYSTEM_KW.items():
        if any(k in t for k in kws):
            return system
    return "general"


def _ensure_repo() -> bool:
    if CACHE.exists():
        return True
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    log.info("cloning MedQuAD (shallow)...")
    try:
        subprocess.run(["git", "clone", "--depth", "1", REPO, str(CACHE)],
                       check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        log.error("clone failed: %s", e.stderr)
        return False


def collect() -> list[dict]:
    if not _ensure_repo():
        return []
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    xmls = list(CACHE.rglob("*.xml"))
    log.info("parsing %d MedQuAD XML files", len(xmls))
    for xml in xmls:
        try:
            root = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        focus = (root.findtext("Focus") or "").strip()
        for qa in root.iter("QAPair"):
            q = (qa.findtext("Question") or "").strip()
            a = (qa.findtext("Answer") or "").strip()
            if len(q) < 5 or len(a) < 30:   # skip empties / licensing-removed
                continue
            qtype = ""
            qel = qa.find("Question")
            if qel is not None:
                qtype = qel.get("qtype", "")
            tags = [t for t in (focus.lower(), qtype) if t]
            records.append({
                "id": Chunk(text=a, source="medquad", type="qa_pair").id,
                "question": q,
                "answer": a[:4000],
                "source": "medquad",
                "type": "qa_pair",
                "body_system": _body_system(f"{focus} {q}"),
                "severity_level": None,
                "tags": tags,
            })
    return records


def main():
    records = collect()
    out = RAW_DIR / "medquad.json"
    out.write_text(json.dumps(records, indent=2), encoding="utf-8")
    log.info("wrote %d QA pairs -> %s", len(records), out)


if __name__ == "__main__":
    main()
