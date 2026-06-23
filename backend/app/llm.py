"""OpenRouter LLM client (OpenAI-compatible chat/completions over httpx)."""

import httpx

from .config import get_settings

SYSTEM_PROMPT = (
    "You are a sentiment-analysis assistant. Given a product review and a set of "
    "similar reviews retrieved from a database, classify the review's sentiment "
    "(Negative, Neutral, or Positive) and explain it in 1-2 sentences. Ground your "
    "explanation in concrete evidence, referencing what similar reviewers said when "
    "relevant. Be concise and do not invent facts not supported by the text."
)


def _build_user_prompt(review: str, retrieved: list[dict]) -> str:
    lines = [f'REVIEW TO ANALYSE:\n"{review}"\n', "SIMILAR REVIEWS RETRIEVED:"]
    if retrieved:
        for i, r in enumerate(retrieved, 1):
            sim = r.get("similarity")
            sim_s = f"{sim:.3f}" if isinstance(sim, (int, float)) else "n/a"
            lines.append(f'  {i}. ({r.get("sentiment", "?")}, sim={sim_s}) "{str(r.get("text",""))[:300]}"')
    else:
        lines.append("  (none)")
    lines.append(
        "\nReturn:\nSentiment: <Negative|Neutral|Positive>\nExplanation: <1-2 sentences>"
    )
    return "\n".join(lines)


async def generate_explanation(review: str, retrieved: list[dict]) -> dict:
    s = get_settings()
    if not s.openrouter_api_key or s.openrouter_api_key.endswith("REPLACE_ME"):
        raise RuntimeError("OPENROUTER_API_KEY is not set in .env")

    payload = {
        "model": s.openrouter_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(review, retrieved)},
        ],
        "temperature": 0.3,
    }
    headers = {
        "Authorization": f"Bearer {s.openrouter_api_key}",
        "Content-Type": "application/json",
        # Optional OpenRouter attribution headers:
        "HTTP-Referer": "http://localhost",
        "X-Title": "RAG Sentiment Agent",
    }
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            f"{s.openrouter_base_url}/chat/completions", json=payload, headers=headers
        )
        resp.raise_for_status()
        data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return {"text": content, "model": s.openrouter_model}
