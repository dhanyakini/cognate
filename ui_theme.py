"""
ui_theme.py — visual identity helpers for the Word Bridge Streamlit app.

Colors, cards, badges, and the signature bridge motif. No learning logic.
"""

from __future__ import annotations

import html

# Hex values match the CSS variables in app.py.
_CATEGORY_COLORS: dict[str, str] = {
    "cognate": "#4C7A5E",
    "false_friend": "#C1272D",
    "unrelated": "#D9A62E",
    "new-word": "#D9A62E",
    "newword": "#D9A62E",
    "new_word": "#D9A62E",
}

_CATEGORY_LABELS: dict[str, str] = {
    "cognate": "Cognate",
    "false_friend": "False friend",
    "unrelated": "New word",
    "new-word": "New word",
    "newword": "New word",
    "new_word": "New word",
}

_INK = "#2B3A67"
_SECONDARY_BG = "#F3E6C4"


def category_color(label: str) -> str:
    """Unknown labels fall back to ink so badges never render unstyled."""
    key = (label or "").strip().lower().replace(" ", "_")
    return _CATEGORY_COLORS.get(key, _INK)


def category_label(label: str) -> str:
    key = (label or "").strip().lower().replace(" ", "_")
    return _CATEGORY_LABELS.get(key, (label or "").replace("_", " "))


def category_tint(label: str, alpha: float = 0.18) -> str:
    """Light wash of the category color (~15–20% opacity) for readable text."""
    hex_color = category_color(label)
    r, g, b = _hex_to_rgb(hex_color)
    return f"rgba({r}, {g}, {b}, {alpha})"


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def script_text(text: str) -> str:
    """Wrap a native-script (or ISO) word in the dedicated typeface class."""
    return f'<span class="script-text">{html.escape(str(text))}</span>'


def render_bridge() -> str:
    """Signature dotted-line bridge with a center node."""
    return (
        '<div class="bridge" style="display:flex;align-items:center;margin:12px 0;">'
        '<div style="flex:1;border-top:2px dotted var(--ink-muted);"></div>'
        '<div style="width:8px;height:8px;border-radius:50%;'
        "background:var(--ink-muted);margin:0 8px;\"></div>"
        '<div style="flex:1;border-top:2px dotted var(--ink-muted);"></div>'
        "</div>"
    )


def render_word_card(
    *,
    heading: str,
    word: str,
    iso: str = "",
    gloss: str = "",
    category: str | None = None,
) -> str:
    """Word-pair card: secondary fill, 12px radius, optional category accent."""
    if category:
        color = category_color(category)
        tint = category_tint(category, 0.16)
        accent = (
            f"border-left:4px solid {color};"
            f"background:color-mix(in srgb, {color} 16%, {_SECONDARY_BG});"
        )
        # Fallback if color-mix unsupported: tint wash over secondary.
        extra_bg = f"background:{tint};"
        style = (
            f"border-radius:12px;padding:18px;{extra_bg}{accent}"
            f"color:var(--ink);"
        )
        cat_attr = html.escape(category)
    else:
        style = (
            f"border-radius:12px;padding:18px;background:{_SECONDARY_BG};"
            "border-left:none;color:var(--ink);"
        )
        cat_attr = "neutral"

    parts = [f'<div class="word-card" data-cat="{cat_attr}" style="{style}">']
    if heading:
        parts.append(
            f'<div class="word-card-heading">{html.escape(heading)}</div>'
        )
    parts.append(f'<div class="word-card-word">{script_text(word)}</div>')
    if iso:
        parts.append(f'<div class="word-card-iso">{script_text(iso)}</div>')
    if gloss:
        parts.append(
            '<div class="word-card-gloss"><strong>Meaning:</strong> '
            f"{html.escape(str(gloss))}</div>"
        )
    parts.append("</div>")
    return "".join(parts)


def render_category_badge(label: str, extra: str = "") -> str:
    color = category_color(label)
    tint = category_tint(label, 0.18)
    text = category_label(label)
    if extra:
        text = f"{text} · {extra}"
    return (
        f'<span class="category-badge" style="'
        f"background:{tint};border:1px solid {color};color:{_INK};"
        f'border-radius:999px;padding:2px 10px;font-size:0.8rem;">'
        f"{html.escape(text)}</span>"
    )


def render_category_banner(label: str, message: str) -> str:
    color = category_color(label)
    tint = category_tint(label, 0.22)
    return (
        f'<div class="category-banner" style="'
        f"background:{tint};border-left:4px solid {color};"
        f"border-radius:12px;padding:14px 18px;margin:8px 0;color:{_INK};"
        f'">'
        f"{render_category_badge(label)}"
        f'<div class="banner-message" style="margin-top:8px;">{message}</div>'
        "</div>"
    )


def render_mixed(text: str) -> str:
    """Body copy that may include native-script words (Noto via .script-text)."""
    return f'<p class="mixed-copy">{script_text(text)}</p>'


def show(fragment: str) -> None:
    """Render a themed HTML fragment. Import streamlit only at call time."""
    import streamlit as st

    st.markdown(fragment, unsafe_allow_html=True)


def nav_button_style(is_active: bool) -> str:
    """CSS declarations for a top-nav button. Reuses --ink / --paper / --ink-muted."""
    if is_active:
        return (
            "background-color: var(--ink) !important;"
            "color: var(--paper) !important;"
            "border: 2px solid var(--ink) !important;"
            "font-weight: 600 !important;"
        )
    return (
        "background-color: transparent !important;"
        "color: var(--ink-muted) !important;"
        "border: 2px solid var(--ink-muted) !important;"
        "font-weight: 500 !important;"
    )

