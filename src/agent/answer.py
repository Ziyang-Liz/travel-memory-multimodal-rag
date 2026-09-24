import os
from typing import List, Optional

from src.retrieval.retrievers import format_results
from src.retrieval.vector_store import SearchResult


SYSTEM_STYLE = """
You are a personalised travel memory assistant.
Answer only using the provided evidence. If the evidence is insufficient, say what is missing.
Use a natural but academic-friendly tone. Keep answers grounded and mention the evidence source briefly.
"""


def _fallback_answer(question: str, evidence: List[SearchResult], route: str) -> str:
    """
    Generate a local grounded answer when no external LLM API key is available.

    """
    if not evidence:
        return (
            "I could not find enough evidence in the travel memory knowledge base. "
            "Please try asking with a clearer place, city, food item, or image description."
        )

    structured_evidence = [
        item for item in evidence
        if item.modality in {"image_caption", "metadata_memory"}
    ]

    text_evidence = [
        item for item in evidence
        if item.modality == "text"
    ]

    display_evidence = structured_evidence[:2]

    if not display_evidence:
        display_evidence = evidence[:2]

    lines = [
        f"Route used: **{route}**",
        "",
        "I found the most relevant travel memory:",
        "",
    ]

    for item in display_evidence:
        city = item.city or item.metadata.get("city", "")
        place = item.place or item.metadata.get("place", "")

        if city and place:
            lines.append(f"**{city} / {place}**")
        elif place:
            lines.append(f"**{place}**")
        elif city:
            lines.append(f"**{city}**")
        else:
            lines.append("**Relevant travel note**")

        image_path = item.source or item.metadata.get("image_path", "")
        caption = item.metadata.get("caption", "")
        related_memory = (
            item.metadata.get("related_memory")
            or item.metadata.get("memory_note")
            or ""
        )
        tags = (
            item.metadata.get("visual_tags")
            or item.metadata.get("tags")
            or ""
        )

        if caption:
            lines.append(f"- Image evidence: {caption}")

        if related_memory:
            lines.append(f"- Memory: {related_memory}")

        if tags:
            lines.append(f"- Tags: {tags}")

        if image_path and image_path.endswith((".jpg", ".jpeg", ".png", ".webp")):
            lines.append(f"- Related image path: `{image_path}`")

        lines.append("")

    if text_evidence:
        source_files = sorted({item.source for item in text_evidence if item.source})
        if source_files:
            lines.append("Supporting text source:")
            for source in source_files[:2]:
                lines.append(f"- `{source}`")
            lines.append("")

    lines.append(
        "This answer is grounded in the retrieved image caption, tags, metadata, "
        "and related travel memory."
    )

    return "\n".join(lines)


def _openai_answer(question: str, evidence: List[SearchResult], route: str, history: str = "") -> Optional[str]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        evidence_text = format_results(evidence)
        prompt = f"""
Question:
{question}

Conversation context:
{history}

Route:
{route}

Evidence:
{evidence_text}

Write a grounded answer. Mention relevant city/place and image path if useful.
"""
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_STYLE},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content
    except Exception as exc:
        return f"OpenAI generation failed, so fallback evidence is shown below.\n\n{exc}\n\n" + _fallback_answer(question, evidence, route)


def generate_answer(question: str, evidence: List[SearchResult], route: str, history: str = "") -> str:
    llm_answer = _openai_answer(question, evidence, route, history)
    if llm_answer:
        return llm_answer
    return _fallback_answer(question, evidence, route)
