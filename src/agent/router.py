import re
from typing import Literal

QueryRoute = Literal["factual", "cross_modal", "multi_hop", "follow_up"]


def contains_whole_word(text: str, word: str) -> bool:
    """Return True only when the keyword appears as a complete word."""
    return re.search(rf"\b{re.escape(word)}\b", text) is not None


def route_query(question: str, conversation_context: str = "") -> QueryRoute:
    """
    Route the user question to the most suitable retrieval strategy.

    The order matters:
    1. Cross-modal questions are checked first because words such as "Hobbit"
       contain the substring "it", which should not be treated as a follow-up.
    2. Multi-hop questions are checked next.
    3. Follow-up questions are only used when conversation history exists.
    4. Otherwise, the question is treated as factual retrieval.
    """
    q = question.lower()

    cross_modal_keywords = [
        "image",
        "photo",
        "picture",
        "caption",
        "visual",
        "shows",
        "showing",
        "look",
        "view",
        "small house",
        "hobbit house",
        "hobbiton",
        "fountain",
        "landmark",
        "food photo",
        "group photo",
    ]

    multi_hop_keywords = [
        "compare",
        "comparison",
        "based on",
        "which city",
        "which place",
        "prefer",
        "preference",
        "more",
        "less",
        "overall",
        "summarise",
        "summarize",
        "summary",
        "recommend",
        "recommendation",
        "similar to",
    ]

    follow_up_words = [
        "that",
        "it",
        "this",
        "those",
        "them",
    ]

    follow_up_phrases = [
        "based on that",
        "as mentioned",
        "the previous one",
        "the last one",
        "next time",
        "less risky",
        "more relaxing",
        "similar option",
    ]

    if any(keyword in q for keyword in cross_modal_keywords):
        return "cross_modal"

    if any(keyword in q for keyword in multi_hop_keywords):
        return "multi_hop"

    if conversation_context:
        if any(contains_whole_word(q, word) for word in follow_up_words):
            return "follow_up"
        if any(phrase in q for phrase in follow_up_phrases):
            return "follow_up"

    return "factual"