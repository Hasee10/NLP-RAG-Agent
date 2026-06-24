"""Phase 6 — grounded, cited, disclaimered answer generation (OpenRouter/Groq).

Uses OPENROUTER_API_KEY + OPENROUTER_BASE_URL from .env (OpenAI-compatible).

Usage:
    python -m medical-rag.generate "What causes high blood pressure?"
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=False)
load_dotenv(ROOT.parent / ".env", override=False)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("generate")

DISCLAIMER = (
    "⚠️ This information is for educational purposes only and does not constitute "
    "medical advice, diagnosis, or treatment. Always consult a licensed healthcare "
    "professional for personal medical decisions."
)

SYSTEM_PROMPT = """You are a grounded medical information assistant.

Rules:
1. Answer ONLY from the provided <sources>. Do not add outside knowledge.
2. Cite each source you use as [1], [2], etc., matching the source numbers.
3. If the sources don't contain enough information to answer, say so clearly.
4. Never diagnose, prescribe, or give personal medical advice.
5. Keep answers concise (3-5 sentences) and accurate.
6. Always end with a brief summary sentence.
"""


def _get_client() -> AsyncOpenAI:
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("GROQ_API_KEY")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


def _get_model() -> str:
    return os.environ.get("OPENROUTER_MODEL", "openrouter/auto")


def _format_sources(chunks: list[dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        src = c.get("source", "unknown")
        ctype = c.get("type", "")
        text = c.get("text", "")[:600]
        lines.append(f"[{i}] ({src} / {ctype}): {text}")
    return "\n\n".join(lines)


async def generate_answer(query: str, chunks: list[dict]) -> dict:
    client = _get_client()
    model = _get_model()

    sources_text = _format_sources(chunks)
    user_msg = f"Question: {query}\n\n<sources>\n{sources_text}\n</sources>"

    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
        max_tokens=512,
    )

    answer = resp.choices[0].message.content.strip()
    citation_ids = [c["id"] for c in chunks if f"[{chunks.index(c)+1}]" in answer]

    return {
        "answer": answer,
        "citations": citation_ids,
        "disclaimer": DISCLAIMER,
        "model": model,
        "sources_used": len(chunks),
    }


def main():
    from retrieve import retrieve

    query = " ".join(sys.argv[1:]) or "What are the symptoms of diabetes?"
    log.info("query: %r", query)
    chunks = retrieve(query, top_k=5)
    log.info("retrieved %d chunks", len(chunks))

    result = asyncio.run(generate_answer(query, chunks))
    print("\n" + "=" * 60)
    print(result["answer"])
    print("\n" + result["disclaimer"])
    print(f"\n[model: {result['model']}]")


if __name__ == "__main__":
    main()
