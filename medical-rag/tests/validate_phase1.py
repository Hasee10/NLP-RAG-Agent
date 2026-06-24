"""Phase 1 validation — assert the collected corpus is well-formed and usable.

Checks: valid JSON, schema-conformant records, id uniqueness, text quality,
per-source field completeness. Exits non-zero on any failure.

Usage:
    python -m medical-rag.tests.validate_phase1
"""

import glob
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from schema import BODY_SYSTEMS, CHUNK_TYPES  # noqa: E402

RAW = ROOT / "data" / "raw"


def main() -> int:
    files = sorted(glob.glob(str(RAW / "*.json")))
    assert files, "no raw data files found — run the collectors first"

    failures: list[str] = []
    ids: dict[str, str] = {}
    total = 0
    by_source = Counter()
    text_lens: list[int] = []

    for f in files:
        name = Path(f).name
        try:
            data = json.load(open(f, encoding="utf-8"))
        except Exception as e:
            failures.append(f"{name}: invalid JSON ({e})")
            continue
        if not isinstance(data, list) or not data:
            failures.append(f"{name}: empty or not a list")
            continue

        for i, r in enumerate(data):
            total += 1
            by_source[r.get("source", "?")] += 1
            loc = f"{name}[{i}]"

            # required fields
            for key in ("id", "text", "source", "type", "body_system"):
                if not r.get(key):
                    failures.append(f"{loc}: missing/empty '{key}'")

            # enum validity
            if r.get("type") not in CHUNK_TYPES:
                failures.append(f"{loc}: bad type {r.get('type')!r}")
            if r.get("body_system") not in BODY_SYSTEMS:
                failures.append(f"{loc}: bad body_system {r.get('body_system')!r}")

            # severity range
            sev = r.get("severity_level")
            if sev is not None and not (isinstance(sev, int) and 1 <= sev <= 5):
                failures.append(f"{loc}: bad severity_level {sev!r}")

            # text quality
            txt = r.get("text") or r.get("answer") or ""
            text_lens.append(len(txt))
            if len(txt.strip()) < 20:
                failures.append(f"{loc}: text too short ({len(txt)} chars)")

            # id uniqueness (true hash collision = different text, same id → failure)
            rid = r.get("id")
            if rid in ids and ids[rid] != txt[:60]:
                failures.append(f"{loc}: hash collision {rid} (different text)")
            ids[rid] = txt[:60]

            # per-type field completeness
            if r.get("type") == "qa_pair" and not r.get("question"):
                failures.append(f"{loc}: qa_pair missing question")
            if r.get("type") == "sentiment_map" and r.get("severity_level") is None:
                failures.append(f"{loc}: sentiment_map missing severity_level")

    # report
    print(f"files validated : {len(files)}")
    print(f"total records   : {total}")
    dupes = total - len(ids)
    print(f"unique ids      : {len(ids)}  (exact duplicates: {dupes} -> removed in Phase 2 dedup)")
    print(f"by source       : {dict(by_source)}")
    if text_lens:
        text_lens.sort()
        print(f"text length     : min={text_lens[0]} median={text_lens[len(text_lens)//2]} max={text_lens[-1]}")

    if failures:
        print(f"\n[FAIL] {len(failures)} issues (showing first 15):")
        for msg in failures[:15]:
            print(f"  - {msg}")
        return 1
    print("\n[PASS] ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
