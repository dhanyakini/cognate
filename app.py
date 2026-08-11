"""
Streamlit language-learning demo: Kannada ↔ Telugu Word Bridge.

Teach-then-test progression: cognate → false_friend → unrelated → test.

Launch:
    streamlit run app.py
"""

from __future__ import annotations

import html

import streamlit as st

from demo_utils import (
    FEATURE_COLS,
    STAGE_LABELS,
    STAGE_ORDER,
    STAGE_THRESHOLD,
    build_test_pool,
    check_meaning_overlap,
    check_word_pair,
    choose_test_type,
    generate_test_question,
    known_target_fields,
    load_data,
    load_model,
    load_teach_unrelated_pairs,
    next_stage,
    pick_question,
    predict_row,
    reconstruct_split,
    render_teach_tip,
    render_tip,
    reverse_lookup,
)
from gloss_lookup import get_gloss_or_prompt
from ui_theme import (
    category_label,
    nav_button_style,
    render_bridge,
    render_category_badge,
    render_category_banner,
    render_mixed,
    render_word_card,
    script_text,
    show,
)

NAV_SCREENS = (
    ("learn", "📖 Learn & Practice"),
    ("checker", "🔍 Check Two Words"),
    ("lookup", "🔎 Look Up a Word"),
)

DIRECTION_UI = {
    "I know Kannada, learning Telugu": "kn_to_te",
    "I know Telugu, learning Kannada": "te_to_kn",
}

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Noto+Serif+Kannada:wght@400;600&family=Noto+Serif+Telugu:wght@400;600&family=Inter:wght@400;500;600&display=swap');

:root {
  --paper: #FBF3E1;
  --ink: #2B3A67;
  --ink-muted: #6B7280;
  --cognate: #4C7A5E;
  --falsefriend: #C1272D;
  --newword: #D9A62E;
  --secondary: #F3E6C4;
}

html, body, .stApp, [data-testid="stAppViewContainer"] {
  background-color: var(--paper);
  color: var(--ink);
  font-family: Inter, sans-serif;
}

h1, h2, h3,
[data-testid="stHeading"] h1,
[data-testid="stHeading"] h2,
[data-testid="stHeading"] h3,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
  font-family: Fraunces, serif !important;
  color: var(--ink) !important;
  letter-spacing: -0.02em;
}

p, label, span, div, li, .stCaption, [data-testid="stCaption"],
[data-testid="stWidgetLabel"], [data-testid="stMetricValue"],
[data-testid="stMetricLabel"] {
  font-family: Inter, sans-serif;
}

button, .stButton button, [data-testid="stBaseButton-primary"],
[data-testid="stBaseButton-secondary"] {
  font-family: Inter, sans-serif !important;
}

.script-text {
  font-family: "Noto Serif Kannada", "Noto Serif Telugu", Fraunces, serif;
  font-weight: 600;
}

.word-card-heading {
  font-family: Inter, sans-serif;
  font-size: 0.82rem;
  font-weight: 500;
  color: var(--ink-muted);
  margin-bottom: 0.35rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.word-card-word {
  font-size: 2rem;
  line-height: 1.25;
  color: var(--ink);
  margin: 0;
}

.word-card-iso {
  font-size: 0.95rem;
  color: var(--ink-muted);
  margin: 0.15rem 0 0.4rem 0;
  font-weight: 400;
}

.word-card-iso .script-text {
  font-weight: 400;
}

.word-card-gloss {
  font-family: Inter, sans-serif;
  font-size: 0.95rem;
  color: var(--ink);
}

.mixed-copy {
  font-family: Inter, sans-serif;
  color: var(--ink);
  line-height: 1.45;
}

.mixed-copy .script-text {
  font-weight: 600;
}

.category-badge {
  font-family: Inter, sans-serif !important;
  font-weight: 600;
}

.st-key-stage_cognate button {
  border-left: 4px solid var(--cognate) !important;
}
.st-key-stage_false_friend button {
  border-left: 4px solid var(--falsefriend) !important;
}
.st-key-stage_unrelated button {
  border-left: 4px solid var(--newword) !important;
}

[data-testid="stSidebar"] {
  background: var(--secondary);
}

.st-key-nav_learn button,
.st-key-nav_checker button,
.st-key-nav_lookup button {
  font-family: Inter, sans-serif !important;
  border-radius: 12px !important;
}
</style>
"""


def _empty_seen_by_stage() -> dict[str, set[str]]:
    return {stage: set() for stage in STAGE_ORDER}


def _empty_cards_seen() -> dict[str, int]:
    return {stage: 0 for stage in STAGE_ORDER}


def _init_state() -> None:
    defaults: dict = {
        "words_learned": 0,
        "streak": 0,
        "best_streak": 0,
        "current_row": None,
        "current_question": None,
        "revealed": False,
        "user_guess": None,
        "model_pred": None,
        "model_proba": None,
        "data_ready": False,
        "direction": "kn_to_te",
        "stage": "cognate",
        "unlocked_stages": {"cognate"},
        "cards_seen_per_stage": _empty_cards_seen(),
        "seen_ids_by_stage": _empty_seen_by_stage(),
        "unlock_banner": None,
        "check_result": None,
        "cw_kn_auto": False,
        "cw_te_auto": False,
        "lookup_results": None,
        "lookup_query": None,
        "screen": "learn",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def _cached_bundle():
    df = reconstruct_split(load_data())
    teach_unrelated = load_teach_unrelated_pairs()
    test_pool = build_test_pool(df, teach_unrelated)
    model = load_model()
    return df, teach_unrelated, test_pool, model


def _reset_learning_progress() -> None:
    st.session_state.words_learned = 0
    st.session_state.streak = 0
    st.session_state.best_streak = 0
    st.session_state.current_row = None
    st.session_state.current_question = None
    st.session_state.revealed = False
    st.session_state.user_guess = None
    st.session_state.model_pred = None
    st.session_state.model_proba = None
    st.session_state.stage = "cognate"
    st.session_state.unlocked_stages = {"cognate"}
    st.session_state.cards_seen_per_stage = _empty_cards_seen()
    st.session_state.seen_ids_by_stage = _empty_seen_by_stage()
    st.session_state.unlock_banner = None


def _label_filter_for_stage(stage: str) -> str | None:
    if stage in {"test", "unrelated"}:
        return None
    return stage


def _pool_for_stage(stage: str):
    if stage == "unrelated":
        return st.session_state.teach_unrelated
    if stage == "test":
        return st.session_state.test_pool
    return st.session_state.df


def _advance(direction: str) -> None:
    stage = st.session_state.stage
    seen = st.session_state.seen_ids_by_stage[stage]
    row = pick_question(
        _pool_for_stage(stage),
        direction,
        seen,
        label_filter=_label_filter_for_stage(stage),
    )
    st.session_state.current_row = row
    st.session_state.current_question = None
    st.session_state.revealed = False
    st.session_state.user_guess = None
    st.session_state.model_pred = None
    st.session_state.model_proba = None
    if row is not None and stage == "test":
        test_type = choose_test_type(row)
        st.session_state.current_question = generate_test_question(
            row,
            direction,
            test_type,
            st.session_state.test_pool,
        )


def _maybe_unlock_next_stage() -> None:
    stage = st.session_state.stage
    if stage == "test":
        return
    if st.session_state.cards_seen_per_stage[stage] < STAGE_THRESHOLD:
        return
    nxt = next_stage(stage)
    if nxt and nxt not in st.session_state.unlocked_stages:
        st.session_state.unlocked_stages.add(nxt)
        st.session_state.unlock_banner = f"{STAGE_LABELS[nxt]} unlocked!"


def _go_to_stage(stage: str) -> None:
    if stage not in st.session_state.unlocked_stages:
        return
    st.session_state.stage = stage
    st.session_state.current_row = None
    st.session_state.current_question = None
    st.session_state.revealed = False
    st.session_state.user_guess = None
    st.session_state.model_pred = None
    st.session_state.model_proba = None
    st.session_state.unlock_banner = None


def _row_has_features(row) -> bool:
    try:
        return all(c in row.index for c in FEATURE_COLS) and all(
            row[c] == row[c] for c in FEATURE_COLS
        )
    except Exception:
        return False


def _teach_category(row) -> str:
    return str(row.get("final_label", st.session_state.stage))


def _render_known_card(
    fields: dict[str, str],
    known_lang: str,
    category: str | None = None,
) -> None:
    show(
        render_word_card(
            heading=f"You know this {known_lang} word",
            word=fields["known_word"],
            iso=fields["known_iso"],
            gloss=fields["known_gloss"],
            category=category,
        )
    )


def _render_target_card(
    fields: dict[str, str],
    target_lang: str,
    category: str | None = None,
) -> None:
    show(
        render_word_card(
            heading=f"Learn this {target_lang} word",
            word=fields["target_word"],
            iso=fields["target_iso"],
            gloss=fields["target_gloss"],
            category=category,
        )
    )


def _teach_panel(row, direction: str, fields: dict[str, str], known_lang: str, target_lang: str) -> None:
    stage = st.session_state.stage
    cat = _teach_category(row)
    show(render_category_badge(cat))
    _render_known_card(fields, known_lang, category=cat)

    if not st.session_state.revealed:
        if st.button("Show me the connection", type="primary", width="stretch"):
            st.session_state.revealed = True
            st.rerun()
        return

    show(render_bridge())
    _render_target_card(fields, target_lang, category=cat)
    show(render_mixed(render_teach_tip(row, direction)))

    if st.button("Next word", type="primary", width="stretch"):
        st.session_state.seen_ids_by_stage[stage].add(str(row["pair_id"]))
        st.session_state.cards_seen_per_stage[stage] += 1
        st.session_state.words_learned += 1
        _maybe_unlock_next_stage()
        st.session_state.revealed = False
        st.session_state.current_row = None
        st.session_state.current_question = None
        st.rerun()


def _test_panel(row, direction: str, fields: dict[str, str], known_lang: str, target_lang: str) -> None:
    q = st.session_state.current_question
    if q is None:
        st.warning("No question generated — advancing.")
        _advance(direction)
        st.rerun()
        return

    mechanic = q["mechanic"]
    gold = str(row["final_label"])

    if mechanic == "guess_meaning":
        show(
            render_word_card(
                heading=f"Target {target_lang} word",
                word=fields["target_word"],
                iso=fields["target_iso"],
                category=None,
            )
        )
        show(
            render_mixed(
                f"Hint from {known_lang}: {fields['known_word']} — {fields['known_gloss']}"
            )
        )
    elif mechanic == "same_or_different":
        _render_known_card(fields, known_lang, category=None)
        show(render_bridge())
        show(
            render_word_card(
                heading=f"Looks like this {target_lang} word",
                word=fields["target_word"],
                iso=fields["target_iso"],
                category=None,
            )
        )
    else:
        _render_known_card(fields, known_lang, category=None)

    show(render_mixed(q["question_text"]))

    if not st.session_state.revealed:
        choice = st.radio(
            "Your answer",
            q["options"],
            index=None,
            key=f"mcq_{row['pair_id']}_{mechanic}",
        )
        if st.button("Submit", type="primary", width="stretch"):
            if choice is None:
                st.warning("Pick an option first.")
                return
            st.session_state.user_guess = choice
            st.session_state.revealed = True
            st.session_state.words_learned += 1
            if choice == q["correct_option"]:
                st.session_state.streak += 1
                st.session_state.best_streak = max(
                    st.session_state.best_streak, st.session_state.streak
                )
            else:
                st.session_state.streak = 0
            if _row_has_features(row):
                pred, proba = predict_row(st.session_state.model, row)
                st.session_state.model_pred = pred
                st.session_state.model_proba = proba
            else:
                st.session_state.model_pred = None
                st.session_state.model_proba = None
            st.rerun()
        return

    guess = st.session_state.user_guess
    correct = q["correct_option"]
    if guess == correct:
        msg = f"Correct — {script_text(str(correct))}."
    else:
        msg = (
            f"Not quite — the answer is {script_text(str(correct))} "
            f"(you chose {script_text(str(guess))})."
        )
    show(render_category_banner(gold, msg))

    _render_known_card(fields, known_lang, category=gold)
    show(render_bridge())
    _render_target_card(fields, target_lang, category=gold)
    show(render_mixed(render_tip(row, direction)))

    pred = st.session_state.model_pred
    proba = st.session_state.model_proba or {}
    if pred is not None:
        conf = 100.0 * float(proba.get(pred, 0.0))
        if pred == gold:
            st.caption(f"Model category guess: {pred} ({conf:.0f}% confidence).")
        else:
            st.caption(
                f"Model category guess: {pred} ({conf:.0f}% confidence); "
                f"gold label is {gold}."
            )
        with st.expander("Why? (similarity scores)"):
            m1, m2, m3 = st.columns(3)
            m1.metric("Orthographic", f"{float(row['orth_sim']):.3f}")
            m2.metric("Phonetic", f"{float(row['phon_sim']):.3f}")
            m3.metric("Semantic", f"{float(row['sem_sim']):.3f}")
        if str(row.get("split", "")) == "train":
            st.caption("Note: this pair was in the model's training set.")

    if st.button("Next word", type="primary", width="stretch"):
        st.session_state.seen_ids_by_stage["test"].add(str(row["pair_id"]))
        st.session_state.cards_seen_per_stage["test"] += 1
        st.session_state.revealed = False
        st.session_state.current_row = None
        st.session_state.current_question = None
        st.session_state.user_guess = None
        st.session_state.model_pred = None
        st.session_state.model_proba = None
        st.rerun()


def render_lookup_screen() -> None:
    st.subheader("Look up a word")
    st.caption("Find known neighbours in the other language.")
    lang_ui = st.radio(
        "Language of your word",
        ["Kannada", "Telugu"],
        horizontal=True,
        key="lu_lang",
        persist_state="session",
    )
    lang = "kn" if lang_ui == "Kannada" else "te"
    word = st.text_input(
        "Word",
        key="lu_word",
        placeholder="Type one word…",
        persist_state="session",
    )

    if st.button("Search", type="primary", width="stretch", key="lu_search"):
        if not (word or "").strip():
            st.warning("Enter a word to look up.")
            return
        with st.spinner("Searching gold + candidate streams…"):
            st.session_state.lookup_results = reverse_lookup(
                word.strip(),
                lang,
                model=st.session_state.model,
            )
            st.session_state.lookup_query = (word.strip(), lang_ui)

    results = st.session_state.get("lookup_results")
    query = st.session_state.get("lookup_query")
    if results is None:
        return

    q_word, q_lang = query if query else ("", "")
    if not results:
        st.info(
            f"No data for **{q_word}** yet. Try **Check two words** below "
            "for a one-off model guess on a specific pair."
        )
        return

    other_lang = "Telugu" if q_lang == "Kannada" else "Kannada"
    st.write(f"**{len(results)}** neighbour(s) in {other_lang}:")
    show(
        render_word_card(
            heading="Searched word",
            word=q_word,
            category=None,
        )
    )
    for hit in results:
        rel = str(hit["relationship"])
        source = hit["source"]
        badge = "Verified" if source == "verified" else "Model guess"
        gloss = hit.get("other_gloss") or "(no gloss)"
        conf = hit.get("confidence")
        extra = badge
        if conf is not None:
            extra = f"{badge} · {100.0 * float(conf):.0f}% conf"
        show(render_bridge())
        show(
            render_word_card(
                heading=f"{category_label(rel)} in {other_lang}",
                word=str(hit["other_word"]),
                gloss=str(gloss),
                category=rel,
            )
        )
        show(render_category_badge(rel, extra=extra))
        if hit.get("gloss_confidence") == "lookup":
            st.caption("(meaning from dictionary lookup, may not be exact)")
        if rel == "unrelated":
            note = check_meaning_overlap(hit.get("sem_sim"))
            if note:
                st.caption(note)


def render_checker_screen() -> None:
    st.subheader("Check two words")
    st.caption("Type a Kannada + Telugu pair for a live model prediction.")
    kn_word = st.text_input("Kannada word", key="cw_kn", persist_state="session")
    show(render_bridge())
    te_word = st.text_input("Telugu word", key="cw_te", persist_state="session")

    if st.button("Auto-fill meanings", width="stretch"):
        kn_g, kn_auto = get_gloss_or_prompt(kn_word, "kn")
        te_g, te_auto = get_gloss_or_prompt(te_word, "te")
        st.session_state.cw_kn_gloss = kn_g or ""
        st.session_state.cw_te_gloss = te_g or ""
        st.session_state.cw_kn_auto = kn_auto
        st.session_state.cw_te_auto = te_auto
        st.session_state.check_result = None

    kn_gloss = st.text_input(
        "Kannada meaning",
        key="cw_kn_gloss",
        placeholder="What does this word mean? (optional -- improves accuracy)",
        persist_state="session",
    )
    if st.session_state.get("cw_kn_auto") and (kn_gloss or "").strip():
        st.caption("Kannada meaning (auto-detected) — edit if wrong.")
    te_gloss = st.text_input(
        "Telugu meaning",
        key="cw_te_gloss",
        placeholder="What does this word mean? (optional -- improves accuracy)",
        persist_state="session",
    )
    if st.session_state.get("cw_te_auto") and (te_gloss or "").strip():
        st.caption("Telugu meaning (auto-detected) — edit if wrong.")

    if st.button("Check", type="primary", width="stretch"):
        if not (kn_word or "").strip() or not (te_word or "").strip():
            st.warning("Enter both a Kannada and a Telugu word.")
            return
        kn_g = (kn_gloss or "").strip() or None
        te_g = (te_gloss or "").strip() or None
        if kn_g is None:
            found, auto = get_gloss_or_prompt(kn_word, "kn")
            if found:
                kn_g = found
                st.session_state.cw_kn_gloss = found
                st.session_state.cw_kn_auto = auto
        if te_g is None:
            found, auto = get_gloss_or_prompt(te_word, "te")
            if found:
                te_g = found
                st.session_state.cw_te_gloss = found
                st.session_state.cw_te_auto = auto
        with st.spinner("Scoring pair…"):
            st.session_state.check_result = check_word_pair(
                kn_word,
                te_word,
                kn_g,
                te_g,
                st.session_state.model,
            )

    result = st.session_state.get("check_result")
    if not result:
        return
    label = result["predicted_label"]
    conf = 100.0 * float(result["confidence"])
    show(
        render_word_card(
            heading="Kannada",
            word=kn_word,
            iso=str(result.get("kn_iso") or ""),
            gloss=str(result.get("kn_gloss_used") or kn_gloss or ""),
            category=label,
        )
    )
    show(render_bridge())
    show(
        render_word_card(
            heading="Telugu",
            word=te_word,
            iso=str(result.get("te_iso") or ""),
            gloss=str(result.get("te_gloss_used") or te_gloss or ""),
            category=label,
        )
    )
    msg = (
        f"Prediction: <strong>{html.escape(category_label(label))}</strong> "
        f"({conf:.0f}% confidence)"
    )
    show(render_category_banner(label, msg))
    c1, c2, c3 = st.columns(3)
    c1.metric("Orth", f"{result['orth_sim']:.3f}")
    c2.metric("Phon", f"{result['phon_sim']:.3f}")
    c3.metric("Sem", f"{result['sem_sim']:.3f}")
    if result["low_confidence"]:
        st.caption(
            "Meaning unknown for one or both words — this prediction relies on "
            "form similarity only and may be less reliable. Add a meaning above "
            "for a better guess."
        )
    elif label == "unrelated":
        note = check_meaning_overlap(result.get("sem_sim"))
        if note:
            st.caption(note)


def _render_top_nav() -> None:
    rules: list[str] = []
    cols = st.columns(3)
    for col, (key, label) in zip(cols, NAV_SCREENS, strict=True):
        is_active = st.session_state.screen == key
        rules.append(f".st-key-nav_{key} button {{ {nav_button_style(is_active)} }}")
        with col:
            if st.button(
                label,
                key=f"nav_{key}",
                width="stretch",
                type="primary" if is_active else "secondary",
            ):
                if st.session_state.screen != key:
                    st.session_state.screen = key
                    st.rerun()
    st.markdown(f"<style>{''.join(rules)}</style>", unsafe_allow_html=True)


def render_learn_screen() -> None:
    st.caption("Learn the patterns first — then test yourself.")

    with st.sidebar:
        st.header("Learning setup")
        direction_ui = st.radio(
            "Direction",
            list(DIRECTION_UI.keys()),
            index=0 if st.session_state.direction == "kn_to_te" else 1,
        )
        direction = DIRECTION_UI[direction_ui]
        if direction != st.session_state.direction:
            st.session_state.direction = direction
            _reset_learning_progress()
            st.rerun()

        st.subheader("Stages")
        for stage in STAGE_ORDER:
            unlocked = stage in st.session_state.unlocked_stages
            label = STAGE_LABELS[stage]
            if unlocked:
                if st.button(
                    label,
                    key=f"stage_{stage}",
                    width="stretch",
                    type="primary" if stage == st.session_state.stage else "secondary",
                ):
                    _go_to_stage(stage)
                    st.rerun()
            else:
                st.button(
                    f"🔒 {label}",
                    key=f"stage_locked_{stage}",
                    width="stretch",
                    disabled=True,
                )

        stage = st.session_state.stage
        if stage != "test":
            seen_n = st.session_state.cards_seen_per_stage[stage]
            st.caption(f"Progress: **{seen_n} / {STAGE_THRESHOLD}** cards in this stage")
        else:
            st.metric("Words practiced (test)", st.session_state.cards_seen_per_stage["test"])
            st.metric("Current streak", st.session_state.streak)
            st.metric("Best streak", st.session_state.best_streak)

        st.metric("Words learned this session", st.session_state.words_learned)
        if st.button("Reset session", width="stretch"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    if st.session_state.unlock_banner:
        st.success(st.session_state.unlock_banner)
        st.session_state.unlock_banner = None

    if st.session_state.current_row is None and not st.session_state.revealed:
        _advance(direction)

    row = st.session_state.current_row
    stage = st.session_state.stage
    known_lang = "Kannada" if direction == "kn_to_te" else "Telugu"
    target_lang = "Telugu" if direction == "kn_to_te" else "Kannada"

    if row is None:
        pretty = STAGE_LABELS.get(stage, stage)
        st.info(f"You've seen all available pairs for **{pretty}** in this direction.")
        nxt = next_stage(stage)
        if nxt:
            if st.button(f"Move on to {STAGE_LABELS[nxt]}", type="primary"):
                st.session_state.unlocked_stages.add(nxt)
                _go_to_stage(nxt)
                st.rerun()
        else:
            st.caption("Reset the session to practice again.")
        return

    fields = known_target_fields(row, direction)
    if stage in {"cognate", "false_friend", "unrelated"}:
        _teach_panel(row, direction, fields, known_lang, target_lang)
    else:
        _test_panel(row, direction, fields, known_lang, target_lang)


def main() -> None:
    st.set_page_config(page_title="Kannada <-> Telugu Word Bridge", layout="centered")
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    _init_state()

    if not st.session_state.data_ready:
        df, teach_unrelated, test_pool, model = _cached_bundle()
        st.session_state.df = df
        st.session_state.teach_unrelated = teach_unrelated
        st.session_state.test_pool = test_pool
        st.session_state.model = model
        st.session_state.data_ready = True

    st.title("Kannada ↔ Telugu Word Bridge")
    _render_top_nav()

    screen = st.session_state.screen
    if screen == "checker":
        render_checker_screen()
    elif screen == "lookup":
        render_lookup_screen()
    else:
        render_learn_screen()


if __name__ == "__main__":
    main()
