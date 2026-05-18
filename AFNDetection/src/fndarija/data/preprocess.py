from __future__ import annotations

import re
import unicodedata


_WS_RE = re.compile(r"\s+")
_DIAC_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")


def normalize_arabic_text(text: str) -> str:
    if not text or not isinstance(text, str):
        return ""
    t = unicodedata.normalize("NFKC", text)
    t = _DIAC_RE.sub("", t)
    t = t.replace("\u0640", "")  # tatwīl
    t = t.replace("\ufeff", "").strip()
    t = _WS_RE.sub(" ", t)
    return t


def compose_document(title: str | None, body: str | None, mode: str) -> str:
    title = normalize_arabic_text(title or "")
    body = normalize_arabic_text(body or "")
    if mode == "title_only":
        return title
    if mode == "text_only":
        return body
    parts = []
    if title:
        parts.append(title)
    if body:
        parts.append(body)
    return " ".join(parts)
