"""Native-script → ISO-15919 (shim; implementation lives in cognate.transliterate)."""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cognate.transliterate import (  # noqa: E402, F401
    SCRIPT_KANNADA,
    SCRIPT_TELUGU,
    TARGET_ISO,
    cache_key,
    load_cache,
    save_cache,
    to_iso,
    to_iso_cached,
)
