"""String similarity helpers (compat shim → features.orthographic)."""

from cognate.features.orthographic import (  # noqa: F401
    first_roman_char,
    levenshtein,
    normalized_similarity,
)
