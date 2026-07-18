"""Native-script → ISO-15919 transliteration via aksharamukha."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import MutableMapping

SCRIPT_KANNADA = "Kannada"
SCRIPT_TELUGU = "Telugu"
TARGET_ISO = "ISO"


def _patch_ast_for_aksharamukha() -> None:
    """aksharamukha does `from ast import Str`, removed in Python 3.14."""
    if not hasattr(ast, "Str"):
        ast.Str = ast.Constant  # type: ignore[attr-defined, misc]


def _process(source_script: str, target: str, text: str) -> str:
    _patch_ast_for_aksharamukha()
    from aksharamukha import transliterate as am

    return str(am.process(source_script, target, text))


def to_iso(text: str, source_script: str) -> str:
    """Transliterate `text` from `source_script` to ISO-15919."""
    if not text:
        return ""
    return _process(source_script, TARGET_ISO, text)


def load_cache(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def save_cache(cache: MutableMapping[str, str], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(dict(cache), f, ensure_ascii=False, indent=0)


def cache_key(source_script: str, text: str) -> str:
    return f"{source_script}\t{text}"


def to_iso_cached(
    text: str,
    source_script: str,
    cache: MutableMapping[str, str],
) -> str:
    """Transliterate with an in-memory (and optionally on-disk) cache."""
    key = cache_key(source_script, text)
    if key in cache:
        return cache[key]
    result = to_iso(text, source_script)
    cache[key] = result
    return result
