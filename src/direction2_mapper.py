from __future__ import annotations

import re
from typing import Iterable


def _normalize(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_\s]+", " ", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _tokenize(text: str) -> list[str]:
    return [t for t in _normalize(text).replace("_", " ").split(" ") if t]


def _build_alias_map(known_labels: set[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for label in known_labels:
        canonical = _normalize(label.replace("_", " "))
        aliases[label] = label
        aliases[canonical] = label

    phrase_alias_candidates = {
        "how are you": ["alright", "good_morning", "good_evening", "good_afternoon", "good_night"],
        "hello": ["alright", "good_morning", "good_evening"],
        "hi": ["alright", "good_morning", "good_evening"],
        "hey": ["alright", "good_morning", "good_evening"],
        "good morning": ["good_morning"],
        "good afternoon": ["good_afternoon"],
        "good evening": ["good_evening"],
        "good night": ["good_night"],
    }

    for phrase, prefs in phrase_alias_candidates.items():
        chosen = next((p for p in prefs if p in known_labels), None)
        if chosen is not None:
            aliases[_normalize(phrase)] = chosen

    return aliases


def map_text_to_direction2_labels(text: str, known_labels: Iterable[str]) -> tuple[list[str], list[str]]:
    known = {str(lb).strip().lower() for lb in known_labels if str(lb).strip()}
    if not known:
        return [], []

    aliases = _build_alias_map(known)
    normalized_full = _normalize(text)
    if normalized_full in aliases:
        return [aliases[normalized_full]], []

    words = _tokenize(text)
    if not words:
        fallback = "alright" if "alright" in known else sorted(known)[0]
        return [fallback], []

    max_parts = max((len(lb.split("_")) for lb in known), default=1)
    out: list[str] = []
    unknown: list[str] = []

    i = 0
    while i < len(words):
        found = None
        for n in range(min(max_parts, len(words) - i), 0, -1):
            phrase = " ".join(words[i:i + n])
            if phrase in aliases:
                found = aliases[phrase]
                i += n
                break

            underscored = phrase.replace(" ", "_")
            if underscored in known:
                found = underscored
                i += n
                break

        if found is not None:
            if not out or out[-1] != found:
                out.append(found)
        else:
            unknown.append(words[i])
            i += 1

    if out:
        return out, unknown

    fallback = "alright" if "alright" in known else sorted(known)[0]
    return [fallback], unknown
