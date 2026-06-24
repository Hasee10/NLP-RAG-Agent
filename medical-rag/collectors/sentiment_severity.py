"""Phase 1 collector — lay-language → clinical-severity map (the sentiment layer).

A curated set mapping how patients describe symptoms emotionally to a clinical
severity score (1=minimal, 5=emergency). This is the second retrieval layer:
it lets the system interpret "my chest feels tight" as severity ~3, not just
match facts. Hand-built because no clean open dataset covers this mapping well.

Usage:
    python -m medical-rag.collectors.sentiment_severity
"""

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from schema import Chunk  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("sentiment")

RAW_DIR = ROOT / "data" / "raw"

# (lay phrase, severity 1-5, clinical framing, body_system)
MAP = [
    ("barely noticeable", 1, "minimal severity, no intervention needed", "general"),
    ("a little uncomfortable", 1, "mild discomfort, self-care appropriate", "general"),
    ("mild discomfort", 2, "mild severity, monitor", "general"),
    ("slightly off", 2, "low-grade symptom, monitor", "general"),
    ("annoying but manageable", 2, "mild, manageable severity", "general"),
    ("noticeable pain", 3, "moderate severity, evaluation advised", "general"),
    ("my chest feels tight", 3, "possible angina; moderate-to-high concern, evaluate", "cardiovascular"),
    ("hard to breathe", 4, "dyspnea; high severity, urgent evaluation", "respiratory"),
    ("can't catch my breath", 4, "acute dyspnea; urgent", "respiratory"),
    ("pounding headache", 3, "moderate headache; evaluate if sudden/severe", "neurological"),
    ("worst headache of my life", 5, "thunderclap headache; emergency, rule out hemorrhage", "neurological"),
    ("dizzy and faint", 3, "presyncope; moderate, evaluate", "neurological"),
    ("stomach is killing me", 3, "significant abdominal pain; evaluate", "gastrointestinal"),
    ("nauseous", 2, "nausea; mild-moderate, monitor", "gastrointestinal"),
    ("throwing up blood", 5, "hematemesis; emergency", "gastrointestinal"),
    ("unbearable pain", 5, "severe pain; urgent evaluation", "general"),
    ("excruciating", 5, "maximal pain severity; emergency", "general"),
    ("crushing chest pain", 5, "possible myocardial infarction; emergency", "cardiovascular"),
    ("numbness on one side", 5, "possible stroke; emergency, call emergency services", "neurological"),
    ("slurred speech", 5, "possible stroke; emergency", "neurological"),
    ("a bit feverish", 2, "low-grade fever; monitor, hydrate", "immune"),
    ("burning up", 4, "high fever; evaluate, especially with other symptoms", "immune"),
    ("exhausted all the time", 3, "persistent fatigue; evaluate underlying cause", "general"),
    ("can't keep food down", 4, "persistent vomiting; risk of dehydration, evaluate", "gastrointestinal"),
    ("rash spreading fast", 4, "rapidly spreading rash; urgent if with fever/swelling", "integumentary"),
    ("swollen and red", 3, "localized inflammation; evaluate for infection", "integumentary"),
    ("heart racing", 3, "palpitations/tachycardia; evaluate", "cardiovascular"),
    ("blacked out", 5, "syncope/loss of consciousness; emergency evaluation", "neurological"),
    ("mild tingling", 2, "paresthesia; mild, monitor", "neurological"),
    ("constant dull ache", 2, "chronic low-grade pain; evaluate if persistent", "musculoskeletal"),
]


def collect() -> list[dict]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    chunks = []
    for phrase, sev, clinical, system in MAP:
        text = f'Patient phrasing "{phrase}" maps to {clinical} (clinical severity {sev}/5).'
        chunks.append(
            Chunk(text=text, source="sentiment_map", type="sentiment_map",
                  body_system=system, severity_level=sev, tags=[phrase]).to_dict()
        )
    return chunks


def main():
    chunks = collect()
    out = RAW_DIR / "sentiment_map.json"
    out.write_text(json.dumps(chunks, indent=2), encoding="utf-8")
    log.info("wrote %d sentiment-severity chunks -> %s", len(chunks), out)


if __name__ == "__main__":
    main()
