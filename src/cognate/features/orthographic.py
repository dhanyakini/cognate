"""Orthographic similarity on ISO-15919 romanizations."""

from __future__ import annotations


def levenshtein(a: str, b: str) -> int:
    """Classic Levenshtein edit distance."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def normalized_similarity(a: str, b: str) -> float:
    """Normalized Levenshtein similarity: 1 - dist / max(len(a), len(b))."""
    if not a and not b:
        return 1.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0
    return 1.0 - levenshtein(a, b) / max_len


def first_roman_char(iso: str) -> str:
    """First alphabetic character of an ISO-15919 string (lowercased)."""
    for ch in iso:
        if ch.isalpha():
            return ch.lower()
    return ""
