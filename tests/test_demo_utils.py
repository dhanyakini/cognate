"""Tests for demo_utils (language-learning Word Bridge)."""

from __future__ import annotations

import pandas as pd

from demo_utils import (
    CATEGORY_COPY,
    SEM_SIMILAR_THRESHOLD,
    TEACH_TIP_TEMPLATES,
    TEACH_UNRELATED_COLS,
    build_distractors,
    check_meaning_overlap,
    generate_test_question,
    known_target_fields,
    load_data,
    load_teach_unrelated_pairs,
    pick_question,
    reconstruct_split,
    render_teach_tip,
    render_tip,
)


def test_reconstruct_split_deterministic() -> None:
    df = load_data()
    a = reconstruct_split(df)
    b = reconstruct_split(df)
    assert list(a["pair_id"]) == list(b["pair_id"])
    assert list(a["split"]) == list(b["split"])


def test_render_tip_swaps_known_target_by_direction() -> None:
    row = {
        "kn_word": "KNWORD",
        "te_word": "TEWORD",
        "en_kn": "water",
        "en_te": "water / liquid",
        "final_label": "cognate",
    }
    tip_kn = render_tip(row, "kn_to_te")
    tip_te = render_tip(row, "te_to_kn")
    assert tip_kn != tip_te
    assert known_target_fields(row, "kn_to_te")["known_word"] == "KNWORD"
    assert known_target_fields(row, "te_to_kn")["known_word"] == "TEWORD"
    assert tip_kn.index("KNWORD") < tip_kn.index("TEWORD")
    assert tip_te.index("TEWORD") < tip_te.index("KNWORD")


def test_render_teach_tip_swaps_known_target_by_direction() -> None:
    # render_tip and render_teach_tip share known_target_fields() for the swap.
    row = {
        "kn_word": "KNWORD",
        "te_word": "TEWORD",
        "en_kn": "water",
        "en_te": "water / liquid",
        "final_label": "cognate",
    }
    tip_kn = render_teach_tip(row, "kn_to_te")
    tip_te = render_teach_tip(row, "te_to_kn")
    assert tip_kn != tip_te
    assert "free word" in tip_kn.lower()
    assert tip_kn.index("KNWORD") < tip_kn.index("TEWORD")
    assert tip_te.index("TEWORD") < tip_te.index("KNWORD")

    ff = {
        "kn_word": "KNWORD",
        "te_word": "TEWORD",
        "en_kn": "known-gloss",
        "en_te": "target-gloss",
        "final_label": "false_friend",
    }
    teach_ff = render_teach_tip(ff, "kn_to_te")
    assert "known-gloss" in teach_ff and "target-gloss" in teach_ff
    assert teach_ff.index("KNWORD") < teach_ff.index("TEWORD")


def test_pick_question_label_filter_cognate_only() -> None:
    df = pd.DataFrame(
        [
            {"pair_id": "C1", "final_label": "cognate"},
            {"pair_id": "F1", "final_label": "false_friend"},
            {"pair_id": "U1", "final_label": "unrelated"},
            {"pair_id": "C2", "final_label": "cognate"},
        ]
    )
    seen: set[str] = set()
    for _ in range(10):
        row = pick_question(df, "kn_to_te", seen, label_filter="cognate")
        if row is None:
            break
        assert row["final_label"] == "cognate"
        assert row["pair_id"] not in {"F1", "U1"}
        seen.add(str(row["pair_id"]))
    assert seen == {"C1", "C2"}


def test_pick_question_returns_none_when_exhausted() -> None:
    df = pd.DataFrame(
        [
            {"pair_id": "A", "final_label": "cognate"},
            {"pair_id": "B", "final_label": "unrelated"},
        ]
    )
    assert pick_question(df, "kn_to_te", {"A", "B"}) is None
    assert pick_question(df, "te_to_kn", {"A", "B"}, label_filter="cognate") is None


def test_category_and_teach_templates_cover_all_three_labels() -> None:
    assert set(CATEGORY_COPY) == {"cognate", "false_friend", "unrelated"}
    assert set(TEACH_TIP_TEMPLATES) == {"cognate", "false_friend", "unrelated"}
    for label in CATEGORY_COPY:
        row = {
            "kn_word": "k",
            "te_word": "t",
            "en_kn": "ek",
            "en_te": "et",
            "final_label": label,
        }
        assert render_tip(row, "kn_to_te")
        assert render_teach_tip(row, "kn_to_te")


def test_load_teach_unrelated_pairs_shared_synset_only() -> None:
    pool = load_teach_unrelated_pairs()
    assert len(pool) == 78
    assert (pool["candidate_source"] == "shared_synset").all()
    assert (pool["final_label"] == "unrelated").all()
    for col in TEACH_UNRELATED_COLS:
        assert col in pool.columns
    # Exhaustion: after seeing all ids, pick_question returns None.
    seen = set(pool["pair_id"].astype(str))
    assert pick_question(pool, "kn_to_te", seen, label_filter=None) is None


def _toy_df() -> pd.DataFrame:
    """Tiny frame with intentional duplicate gloss/word text for distractor tests."""
    return pd.DataFrame(
        [
            {
                "pair_id": "C1",
                "kn_word": "ನೀರು",
                "te_word": "నీరు",
                "kn_iso": "niiru",
                "te_iso": "niiru",
                "en_kn": "water",
                "en_te": "water",
                "final_label": "cognate",
            },
            {
                "pair_id": "D1",
                "kn_word": "ಹಾಲು",
                "te_word": "పాలు",
                "kn_iso": "haalu",
                "te_iso": "paalu",
                "en_kn": "milk",
                "en_te": "milk",
                "final_label": "cognate",
            },
            {
                "pair_id": "D2",
                "kn_word": "ಬೆಂಕಿ",
                "te_word": "నిప్పు",
                "kn_iso": "beṅki",
                "te_iso": "nippu",
                "en_kn": "fire",
                "en_te": "fire",
                "final_label": "unrelated",
            },
            {
                "pair_id": "D3",
                "kn_word": "ಮನೆ",
                "te_word": "ఇల్లు",
                "kn_iso": "mane",
                "te_iso": "illu",
                "en_kn": "house",
                "en_te": "house",
                "final_label": "unrelated",
            },
            {
                "pair_id": "D4",
                "kn_word": "ಸೂರ್ಯ",
                "te_word": "సూర్యుడు",
                "kn_iso": "suurya",
                "te_iso": "suuryuḍu",
                "en_kn": "sun",
                "en_te": "sun",
                "final_label": "cognate",
            },
            {
                # Same target gloss as C1 ("water") — must never appear as a
                # distractor when C1's correct answer is also "water".
                "pair_id": "DUP",
                "kn_word": "ಜಲ",
                "te_word": "జలము",
                "kn_iso": "jala",
                "te_iso": "jalamu",
                "en_kn": "water",
                "en_te": "water",
                "final_label": "cognate",
            },
            {
                "pair_id": "FF1",
                "kn_word": "ಕೂತಿರು",
                "te_word": "కూతురు",
                "kn_iso": "kuutiru",
                "te_iso": "kuuturu",
                "en_kn": "to sit",
                "en_te": "daughter",
                "final_label": "false_friend",
            },
        ]
    )


def test_build_distractors_skips_duplicate_correct_text() -> None:
    df = _toy_df()
    correct = df[df["pair_id"] == "C1"].iloc[0]
    for _ in range(20):
        options = build_distractors(df, correct, n=3, pool="gloss", direction="kn_to_te")
        assert options.count("water") == 1
        assert len(options) == len(set(options))
        assert "water" in options


def test_generate_test_question_each_type_has_one_correct() -> None:
    df = _toy_df()
    cognate = df[df["pair_id"] == "C1"].iloc[0]
    ff = df[df["pair_id"] == "FF1"].iloc[0]
    unrelated = df[df["pair_id"] == "D2"].iloc[0]

    q_mean = generate_test_question(cognate, "kn_to_te", "guess_meaning", df)
    assert q_mean["mechanic"] == "guess_meaning"
    assert q_mean["options"].count(q_mean["correct_option"]) == 1
    assert q_mean["correct_option"] == "water"

    q_same_ff = generate_test_question(ff, "kn_to_te", "same_or_different", df)
    assert q_same_ff["correct_option"] == "Different meaning"
    assert set(q_same_ff["options"]) == {"Same meaning", "Different meaning"}
    assert q_same_ff["options"].count("Different meaning") == 1

    q_same_cog = generate_test_question(cognate, "kn_to_te", "same_or_different", df)
    assert q_same_cog["correct_option"] == "Same meaning"
    assert q_same_cog["options"].count("Same meaning") == 1

    q_word = generate_test_question(unrelated, "kn_to_te", "guess_word", df)
    assert q_word["mechanic"] == "guess_word"
    assert q_word["correct_option"] == "నిప్పు"
    assert q_word["options"].count("నిప్పు") == 1
    assert len(q_word["options"]) == len(set(q_word["options"]))


def test_check_meaning_overlap_none_when_sem_sim_missing() -> None:
    assert check_meaning_overlap(None) is None


def test_check_meaning_overlap_none_below_threshold() -> None:
    assert SEM_SIMILAR_THRESHOLD == 0.75
    assert check_meaning_overlap(0.74) is None
    assert check_meaning_overlap(0.5) is None  # imputed midpoint must not fire


def test_check_meaning_overlap_note_at_or_above_threshold() -> None:
    note = check_meaning_overlap(0.75)
    assert note is not None
    assert "semantic similarity: 75%" in note
    assert "not a cognate" in note.lower()
    high = check_meaning_overlap(1.0)
    assert high is not None
    assert "semantic similarity: 100%" in high
