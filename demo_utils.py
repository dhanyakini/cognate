"""
demo_utils.py — helpers for the Kannada ↔ Telugu Word Bridge learning demo.

Reuses data/gold.csv, data/features.csv, and data/model.joblib.
Does not retrain or recompute embeddings.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
GOLD_PATH = ROOT / "data" / "gold.csv"
FEATURES_PATH = ROOT / "data" / "features.csv"
MODEL_PATH = ROOT / "data" / "model.joblib"
FEATURE_COLS = ["orth_sim", "phon_sim", "sem_sim"]

# Must match src/cognate/model.py exactly.
TEST_SIZE = 0.3
RANDOM_STATE = 42

STAGE_ORDER = ["cognate", "false_friend", "unrelated", "test"]
STAGE_THRESHOLD = 6  # teach cards before next stage unlocks (tune for demos)
SEM_SIMILAR_THRESHOLD = 0.75  # real (non-imputed) sem_sim for the meaning-overlap note

STAGE_LABELS = {
    "cognate": "🎉 Cognates",
    "false_friend": "⚠️ False friends",
    "unrelated": "📘 Unrelated",
    "test": "🧪 Test",
}

CATEGORY_COPY: dict[str, dict[str, str]] = {
    "cognate": {
        "emoji": "🎉",
        "headline": "Free word!",
        "tip_template": (
            "Great instinct — {known_word} and {target_word} both relate to "
            "'{gloss_hint}'. You can often guess words like this."
        ),
    },
    "false_friend": {
        "emoji": "⚠️",
        "headline": "Watch out — false friend",
        "tip_template": (
            "Careful — {target_word} looks like {known_word} but actually means "
            "'{target_gloss}', not '{known_gloss}'. Don't assume!"
        ),
    },
    "unrelated": {
        "emoji": "📘",
        "headline": "New word to learn",
        "tip_template": (
            "No shortcut here — {known_word} and {target_word} aren't connected "
            "in form or meaning. Worth memorizing as new vocabulary."
        ),
    },
}

# Declarative teach-mode tips (no guess framing).
# render_tip and render_teach_tip both use known_target_fields() for direction swap.
TEACH_TIP_TEMPLATES: dict[str, str] = {
    "cognate": (
        "This one's a free word — {known_word} and {target_word} both relate to "
        "'{gloss_hint}'. Notice how similar they look."
    ),
    "false_friend": (
        "Watch this trap — {known_word} and {target_word} look alike, but "
        "{known_word} means '{known_gloss}' while {target_word} means "
        "'{target_gloss}'. Don't mix these up."
    ),
    "unrelated": (
        "No shortcut here — {known_word} and {target_word} aren't connected. "
        "Just a new word to add to your vocabulary."
    ),
}


def load_data(
    gold_path: str | Path = GOLD_PATH,
    features_path: str | Path = FEATURES_PATH,
) -> pd.DataFrame:
    """Merge gold + features on pair_id (inner join)."""
    gold = pd.read_csv(gold_path)
    feats = pd.read_csv(features_path)
    if "pair_id" not in gold.columns or "pair_id" not in feats.columns:
        raise ValueError("gold and features must both have pair_id")
    df = gold.merge(feats, on="pair_id", how="inner", suffixes=("", "_feat"))
    if "label" not in df.columns and "final_label" in df.columns:
        df["label"] = df["final_label"]
    return df.reset_index(drop=True)


TEACH_UNRELATED_COLS = [
    "pair_id",
    "kn_word",
    "te_word",
    "kn_iso",
    "te_iso",
    "en_kn",
    "en_te",
]


def load_teach_unrelated_pairs(
    gold_path: str | Path = GOLD_PATH,
) -> pd.DataFrame:
    """
    Same-meaning, non-cognate vocabulary pairs for the unrelated teach stage.

    Stream-A gold rows only (``candidate_source == "shared_synset"`` and
    ``final_label == "unrelated"``). Random/form_similar negatives are excluded
    — those do not share meaning and must not be taught as translations.

    Returns the teaching schema columns plus ``candidate_source`` /
    ``final_label`` (always ``shared_synset`` / ``unrelated``) so tip rendering
    and pool audits keep working without a separate label override.
    """
    gold = pd.read_csv(gold_path)
    pool = gold[
        (gold["candidate_source"] == "shared_synset")
        & (gold["final_label"] == "unrelated")
    ].copy()
    keep = TEACH_UNRELATED_COLS + ["candidate_source", "final_label"]
    missing = [c for c in keep if c not in pool.columns]
    if missing:
        raise ValueError(f"gold missing columns for teach unrelated pool: {missing}")
    return pool[keep].reset_index(drop=True)


def reconstruct_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Add a ``split`` column ("train"/"test") matching model.py training.

    Copied call from ``cognate.model.split_train_test``:
        train_test_split(df, test_size=0.3, random_state=42, stratify=df["label"])
    """
    if "label" not in df.columns:
        raise ValueError("df must have a 'label' column for stratified split")
    out = df.copy()
    train_df, test_df = train_test_split(
        out,
        test_size=test_size,
        random_state=random_state,
        stratify=out["label"],
    )
    del test_df
    train_ids = set(train_df["pair_id"].astype(str))
    out["split"] = out["pair_id"].astype(str).map(
        lambda pid: "train" if pid in train_ids else "test"
    )
    return out.reset_index(drop=True)


def load_model(path: str | Path = MODEL_PATH) -> Any:
    return joblib.load(path)


def predict_row(
    model: Any,
    row: pd.Series | dict[str, Any],
) -> tuple[str, dict[str, float]]:
    if isinstance(row, pd.Series):
        values = row[FEATURE_COLS].to_numpy(dtype=float).reshape(1, -1)
    else:
        values = [[float(row[c]) for c in FEATURE_COLS]]
    pred = str(model.predict(values)[0])
    proba_arr = model.predict_proba(values)[0]
    proba = {
        str(cls): float(p) for cls, p in zip(model.classes_, proba_arr, strict=True)
    }
    return pred, proba


# Midpoint imputation matching cognate.build.SEMANTIC_IMPUTE_VALUE.
_SEMANTIC_IMPUTE = 0.5
_SEMANTIC_CACHE_PATH = ROOT / "data" / "cache" / "semantic_embeddings.json"
_semantic_cache: dict[str, list[float]] | None = None


def _get_semantic_cache() -> dict[str, list[float]]:
    """Lazy on-disk embedding cache (same file featurize uses)."""
    global _semantic_cache
    if _semantic_cache is None:
        from cognate.features.semantic import load_cache

        _semantic_cache = load_cache(_SEMANTIC_CACHE_PATH)
    return _semantic_cache


def check_word_pair(
    kn_word: str,
    te_word: str,
    kn_gloss: str | None,
    te_gloss: str | None,
    model: Any,
) -> dict[str, Any]:
    """
    Live cognate / false_friend / unrelated prediction for a typed pair.

    Reuses transliterate + orthographic / phonetic / semantic feature modules
    and the trained ``model.joblib`` — no reimplementation of feature logic.
    """
    from cognate.features.orthographic import normalized_similarity
    from cognate.features.phonetic import phonetic_similarity
    from cognate.features.semantic import semantic_similarity
    from cognate.transliterate import SCRIPT_KANNADA, SCRIPT_TELUGU, to_iso

    kn = (kn_word or "").strip()
    te = (te_word or "").strip()
    kn_g = (kn_gloss or "").strip() or None
    te_g = (te_gloss or "").strip() or None

    kn_iso = to_iso(kn, SCRIPT_KANNADA)
    te_iso = to_iso(te, SCRIPT_TELUGU)
    orth_sim = float(normalized_similarity(kn_iso, te_iso))
    try:
        phon_sim = float(phonetic_similarity(kn_iso, te_iso))
    except ValueError:
        phon_sim = float(phonetic_similarity(kn_iso, te_iso, backend="nw"))

    low_confidence = kn_g is None or te_g is None
    if low_confidence:
        sem_sim = _SEMANTIC_IMPUTE
    else:
        sem = semantic_similarity(kn_g, te_g, cache=_get_semantic_cache())
        sem_sim = float(sem) if sem is not None else _SEMANTIC_IMPUTE
        if sem is None:
            low_confidence = True

    features = {
        "orth_sim": orth_sim,
        "phon_sim": phon_sim,
        "sem_sim": sem_sim,
    }
    pred, proba = predict_row(model, features)
    return {
        "predicted_label": pred,
        "confidence": float(proba.get(pred, 0.0)),
        "orth_sim": orth_sim,
        "phon_sim": phon_sim,
        "sem_sim": sem_sim,
        "low_confidence": low_confidence,
        "kn_gloss_used": kn_g,
        "te_gloss_used": te_g,
        "kn_iso": kn_iso,
        "te_iso": te_iso,
        "proba": proba,
    }


STREAM_A_GLOSSED_PATH = ROOT / "data" / "candidates" / "stream_a_glossed.csv"
STREAM_B_GLOSSED_PATH = ROOT / "data" / "candidates" / "stream_b_glossed.csv"

_lookup_tables: dict[str, pd.DataFrame] | None = None


def reset_lookup_tables() -> None:
    """Clear cached gold/stream tables (tests)."""
    global _lookup_tables
    _lookup_tables = None


def check_meaning_overlap(sem_sim: float | None) -> str | None:
    """
    Note for same-meaning but historically unrelated pairs.

    Callers should pass ``None`` when ``sem_sim`` was imputed (no real glosses).
    Does not look at the relationship label — wire it only for ``unrelated``.
    """
    if sem_sim is None:
        return None
    if float(sem_sim) < SEM_SIMILAR_THRESHOLD:
        return None
    pct = int(round(100.0 * float(sem_sim)))
    return (
        "Even though these aren't historically related (not a cognate), "
        f"they seem to mean similar things (semantic similarity: {pct}%) -- "
        "could be a handy vocabulary pair to learn, just not a 'freebie' from "
        "shared origin."
    )


def _clean_cell(value: object) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _load_lookup_tables() -> dict[str, pd.DataFrame]:
    """Lazy-load gold + glossed stream CSVs once."""
    global _lookup_tables
    if _lookup_tables is not None:
        return _lookup_tables
    cols = ["kn_word", "te_word", "en_kn", "en_te", "kn_iso", "te_iso"]
    tables: dict[str, pd.DataFrame] = {}
    gold = pd.read_csv(GOLD_PATH)
    keep_gold = [c for c in cols + ["final_label", "pair_id"] if c in gold.columns]
    tables["gold"] = gold[keep_gold].copy()
    for key, path in (
        ("stream_a", STREAM_A_GLOSSED_PATH),
        ("stream_b", STREAM_B_GLOSSED_PATH),
    ):
        if not path.exists():
            tables[key] = pd.DataFrame(columns=cols)
            continue
        df = pd.read_csv(path)
        keep = [c for c in cols if c in df.columns]
        tables[key] = df[keep].copy()
    if FEATURES_PATH.exists():
        tables["features"] = pd.read_csv(FEATURES_PATH)
    else:
        tables["features"] = pd.DataFrame()
    _lookup_tables = tables
    return tables


def _sem_sim_by_pair_id(features: pd.DataFrame) -> dict[str, float | None]:
    """Map pair_id → real sem_sim, or None when the value was imputed/missing."""
    out: dict[str, float | None] = {}
    if features is None or features.empty or "pair_id" not in features.columns:
        return out
    for _, row in features.iterrows():
        pid = str(row["pair_id"]).strip()
        imputed = str(row.get("semantic_imputed", "")).strip().lower() in {
            "true",
            "1",
            "yes",
        }
        if imputed or "sem_sim" not in row.index or pd.isna(row["sem_sim"]):
            out[pid] = None
        else:
            out[pid] = float(row["sem_sim"])
    return out


def _gloss_with_confidence(
    row_gloss: str,
    other_word: str,
    other_lang: str,
) -> tuple[str | None, str]:
    """Prefer the row's own gloss; else gloss_lookup.find_gloss; else unavailable."""
    native = (row_gloss or "").strip() or None
    if native:
        return native, "dataset"
    from gloss_lookup import find_gloss

    filled = find_gloss(other_word, other_lang)
    filled = (filled or "").strip() or None
    if filled:
        return filled, "lookup"
    return None, "unavailable"


def reverse_lookup(
    word: str,
    lang: str,
    model: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Find other-language neighbours of ``word`` across gold + glossed streams.

    Gold hits are returned with ``source="verified"`` and ``relationship`` =
    ``final_label``. Stream-only hits get a live model prediction with
    ``source="model_predicted"``. Duplicate pairs prefer the gold row.
    """
    word = (word or "").strip()
    if not word:
        return []
    if lang not in {"kn", "te"}:
        raise ValueError(f"lang must be 'kn' or 'te', got {lang!r}")

    query_col = "kn_word" if lang == "kn" else "te_word"
    other_col = "te_word" if lang == "kn" else "kn_word"
    other_gloss_col = "en_te" if lang == "kn" else "en_kn"
    query_gloss_col = "en_kn" if lang == "kn" else "en_te"

    tables = _load_lookup_tables()
    other_lang = "te" if lang == "kn" else "kn"
    sem_by_id = _sem_sim_by_pair_id(tables.get("features", pd.DataFrame()))
    # pair_key -> result dict (gold overwrites / blocks stream)
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}

    gold = tables["gold"]
    gold_hits = gold[gold[query_col].astype(str).str.strip() == word]
    for _, row in gold_hits.iterrows():
        kn = _clean_cell(row["kn_word"])
        te = _clean_cell(row["te_word"])
        if not kn or not te:
            continue
        other_word = _clean_cell(row[other_col])
        other_gloss, gloss_confidence = _gloss_with_confidence(
            _clean_cell(row[other_gloss_col]),
            other_word,
            other_lang,
        )
        pid = _clean_cell(row.get("pair_id", ""))
        by_pair[(kn, te)] = {
            "other_word": other_word,
            "other_gloss": other_gloss,
            "gloss_confidence": gloss_confidence,
            "relationship": _clean_cell(row.get("final_label", "")) or "unrelated",
            "source": "verified",
            "query_word": word,
            "query_gloss": _clean_cell(row[query_gloss_col]) or None,
            "sem_sim": sem_by_id.get(pid),
        }

    model_obj = model
    for stream_key in ("stream_a", "stream_b"):
        stream = tables[stream_key]
        if stream.empty or query_col not in stream.columns:
            continue
        hits = stream[stream[query_col].astype(str).str.strip() == word]
        for _, row in hits.iterrows():
            kn = _clean_cell(row["kn_word"])
            te = _clean_cell(row["te_word"])
            if not kn or not te:
                continue
            if (kn, te) in by_pair:
                continue  # gold (or earlier stream) already claimed this pair
            if model_obj is None:
                model_obj = load_model()
            other_word = te if lang == "kn" else kn
            other_gloss, gloss_confidence = _gloss_with_confidence(
                _clean_cell(row[other_gloss_col]),
                other_word,
                other_lang,
            )
            kn_g = _clean_cell(row.get("en_kn", "")) or None
            te_g = _clean_cell(row.get("en_te", "")) or None
            if lang == "kn":
                te_g = other_gloss
            else:
                kn_g = other_gloss
            scored = check_word_pair(kn, te, kn_g, te_g, model_obj)
            by_pair[(kn, te)] = {
                "other_word": other_word,
                "other_gloss": other_gloss,
                "gloss_confidence": gloss_confidence,
                "relationship": scored["predicted_label"],
                "source": "model_predicted",
                "confidence": scored["confidence"],
                "low_confidence": scored["low_confidence"],
                "query_word": word,
                "query_gloss": _clean_cell(row[query_gloss_col]) or None,
                # Imputed 0.5 is not evidence — callers must see None.
                "sem_sim": None if scored["low_confidence"] else scored["sem_sim"],
            }

    # Stable-ish order: verified first, then model guesses.
    results = list(by_pair.values())
    results.sort(key=lambda r: (0 if r["source"] == "verified" else 1, r["other_word"]))
    return results


def known_target_fields(
    row: pd.Series | dict[str, Any],
    direction: str,
) -> dict[str, str]:
    """
    Shared known/target column swap used by both render_tip and render_teach_tip.

    (Single helper — do not re-implement this swap in either tip renderer.)
    """
    if direction not in {"kn_to_te", "te_to_kn"}:
        raise ValueError(f"unknown direction: {direction!r}")
    get = row.get if isinstance(row, dict) else row.__getitem__
    if direction == "kn_to_te":
        return {
            "known_word": str(get("kn_word")),
            "known_iso": str(get("kn_iso")),
            "known_gloss": str(get("en_kn")),
            "target_word": str(get("te_word")),
            "target_iso": str(get("te_iso")),
            "target_gloss": str(get("en_te")),
        }
    return {
        "known_word": str(get("te_word")),
        "known_iso": str(get("te_iso")),
        "known_gloss": str(get("en_te")),
        "target_word": str(get("kn_word")),
        "target_iso": str(get("kn_iso")),
        "target_gloss": str(get("en_kn")),
    }


def _format_tip(
    template: str,
    row: pd.Series | dict[str, Any],
    direction: str,
) -> str:
    fields = known_target_fields(row, direction)
    gloss_hint = fields["known_gloss"] or fields["target_gloss"]
    return template.format(
        known_word=fields["known_word"],
        target_word=fields["target_word"],
        known_gloss=fields["known_gloss"],
        target_gloss=fields["target_gloss"],
        gloss_hint=gloss_hint,
    )


def render_tip(row: pd.Series | dict[str, Any], direction: str) -> str:
    """Guess-framed tip (test mode). Uses known_target_fields for direction swap."""
    get = row.get if isinstance(row, dict) else row.__getitem__
    label = str(get("final_label")).strip()
    copy = CATEGORY_COPY.get(label)
    if copy is None:
        return (
            f"Label '{label}' is outside the three learning categories — "
            "treat this pair carefully."
        )
    return _format_tip(copy["tip_template"], row, direction)


def render_teach_tip(row: pd.Series | dict[str, Any], direction: str) -> str:
    """Declarative teach tip. Uses the same known_target_fields swap as render_tip."""
    get = row.get if isinstance(row, dict) else row.__getitem__
    label = str(get("final_label")).strip()
    template = TEACH_TIP_TEMPLATES.get(label)
    if template is None:
        return (
            f"Label '{label}' is outside the three learning categories — "
            "treat this pair carefully."
        )
    return _format_tip(template, row, direction)


def pick_question(
    df: pd.DataFrame,
    direction: str,
    seen_ids: set[str],
    label_filter: str | None = None,
) -> pd.Series | None:
    """
    Return one unseen row, or None if the pool is exhausted.

    ``direction`` does not filter rows — it only informs the UI layer.
    ``label_filter`` restricts to ``final_label == label_filter`` when set
    (teach stages); ``None`` allows any label (test stage).
    """
    if direction not in {"kn_to_te", "te_to_kn"}:
        raise ValueError(f"unknown direction: {direction!r}")
    pool = df.copy()
    pool["_pid"] = pool["pair_id"].astype(str)
    pool = pool[~pool["_pid"].isin(seen_ids)]
    if label_filter is not None:
        pool = pool[pool["final_label"] == label_filter]
    if pool.empty:
        return None
    return pool.sample(n=1, random_state=None).iloc[0]


def next_stage(stage: str) -> str | None:
    """Unknown stage names return None rather than raising."""
    try:
        idx = STAGE_ORDER.index(stage)
    except ValueError:
        return None
    if idx + 1 >= len(STAGE_ORDER):
        return None
    return STAGE_ORDER[idx + 1]


def build_test_pool(
    df: pd.DataFrame,
    teach_unrelated: pd.DataFrame,
) -> pd.DataFrame:
    """
    Mixed test pool: gold cognate + false_friend, plus same-meaning
    Stream-A unrelated pairs from ``load_teach_unrelated_pairs``.
    """
    cf = df[df["final_label"].isin(["cognate", "false_friend"])].copy()
    u_ids = set(teach_unrelated["pair_id"].astype(str))
    un = df[
        (df["final_label"] == "unrelated")
        & (df["pair_id"].astype(str).isin(u_ids))
    ].copy()
    return pd.concat([cf, un], ignore_index=True).reset_index(drop=True)


def choose_test_type(row: pd.Series | dict[str, Any]) -> str:
    """
    Map a row's category to a test mechanic.

    Cognates are split 50/50 between meaning-MCQ and same/different so the
    discrimination question isn't trivially always "Different meaning".
    """
    get = row.get if isinstance(row, dict) else row.__getitem__
    label = str(get("final_label")).strip()
    if label == "false_friend":
        return "same_or_different"
    if label == "unrelated":
        return "guess_word"
    if label == "cognate":
        return "guess_meaning" if random.random() < 0.5 else "same_or_different"
    raise ValueError(f"no test type for label {label!r}")


def build_distractors(
    df: pd.DataFrame,
    correct_row: pd.Series | dict[str, Any],
    n: int = 3,
    pool: str = "gloss",
    direction: str = "kn_to_te",
) -> list[str]:
    """
    Build a shuffled MCQ option list: correct answer + ``n`` distractors.

    ``pool="gloss"`` samples other rows' target gloss; ``pool="word"`` samples
    other rows' target word (direction-aware via ``known_target_fields``).
    Duplicate text vs the correct answer (or other options) is skipped.
    """
    if pool not in {"gloss", "word"}:
        raise ValueError(f"unknown distractor pool: {pool!r}")
    if direction not in {"kn_to_te", "te_to_kn"}:
        raise ValueError(f"unknown direction: {direction!r}")

    fields = known_target_fields(correct_row, direction)
    correct = fields["target_gloss"] if pool == "gloss" else fields["target_word"]
    correct = str(correct).strip()
    key = "target_gloss" if pool == "gloss" else "target_word"

    get = correct_row.get if isinstance(correct_row, dict) else correct_row.__getitem__
    correct_pid = str(get("pair_id"))

    others = df[df["pair_id"].astype(str) != correct_pid]
    distractors: list[str] = []
    if not others.empty:
        for _, r in others.sample(frac=1.0, random_state=None).iterrows():
            text = str(known_target_fields(r, direction)[key]).strip()
            if not text or text == correct or text in distractors:
                continue
            distractors.append(text)
            if len(distractors) >= n:
                break

    options = distractors + [correct]
    random.shuffle(options)
    return options


def generate_test_question(
    row: pd.Series | dict[str, Any],
    direction: str,
    test_type: str,
    full_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Build a multiple-choice test item for ``test_type``.

    Returns ``prompt_word``, ``prompt_gloss``, ``question_text``, ``options``,
    ``correct_option``, ``mechanic``.
    """
    if test_type not in {"guess_meaning", "same_or_different", "guess_word"}:
        raise ValueError(f"unknown test_type: {test_type!r}")
    if direction not in {"kn_to_te", "te_to_kn"}:
        raise ValueError(f"unknown direction: {direction!r}")

    fields = known_target_fields(row, direction)
    get = row.get if isinstance(row, dict) else row.__getitem__
    label = str(get("final_label")).strip()
    target_lang = "Telugu" if direction == "kn_to_te" else "Kannada"

    if test_type == "guess_meaning":
        options = build_distractors(
            full_df, row, n=3, pool="gloss", direction=direction
        )
        return {
            "prompt_word": fields["target_word"],
            "prompt_gloss": fields["known_gloss"],
            "question_text": (
                f"You know {fields['known_word']} means '{fields['known_gloss']}'. "
                f"What does {fields['target_word']} mean?"
            ),
            "options": options,
            "correct_option": fields["target_gloss"],
            "mechanic": "guess_meaning",
        }

    if test_type == "same_or_different":
        if label == "false_friend":
            correct = "Different meaning"
        elif label == "cognate":
            correct = "Same meaning"
        else:
            raise ValueError(
                "same_or_different requires cognate or false_friend rows, "
                f"got {label!r}"
            )
        options = ["Same meaning", "Different meaning"]
        random.shuffle(options)
        return {
            "prompt_word": fields["known_word"],
            "prompt_gloss": fields["known_gloss"],
            "question_text": (
                f"{fields['known_word']} looks like {fields['target_word']}. "
                "Do they mean the same thing?"
            ),
            "options": options,
            "correct_option": correct,
            "mechanic": "same_or_different",
        }

    # guess_word
    options = build_distractors(full_df, row, n=3, pool="word", direction=direction)
    return {
        "prompt_word": fields["known_word"],
        "prompt_gloss": fields["known_gloss"],
        "question_text": (
            f"{fields['known_word']} means '{fields['known_gloss']}'. "
            f"Which {target_lang} word means the same?"
        ),
        "options": options,
        "correct_option": fields["target_word"],
        "mechanic": "guess_word",
    }
