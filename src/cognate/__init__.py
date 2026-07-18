"""Kannada–Telugu cognate / false-friend utilities."""

CSV_HEADER = [
    "pair_id",
    "kn_word",
    "te_word",
    "kn_iso",
    "te_iso",
    "synset_id",
    "gloss",
    "candidate_source",
    "label",
    "origin",
    "annotator",
    "notes",
]


def bilingual_gloss(kn_gloss: str, te_gloss: str) -> str:
    """Canonical bilingual gloss: kn: <...> || te: <...>."""
    return f"kn: {kn_gloss} || te: {te_gloss}"
