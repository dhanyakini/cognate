# Annotation Guidelines — Kannada–Telugu Cognate / False-Friend Dataset

**Version:** 0.1 (pilot)  **Task:** label each candidate word pair as `cognate`, `false_friend`, or `unrelated`.

These guidelines are meant to be piloted on ~30–50 pairs, revised after an inter-annotator agreement (kappa) check, and only then applied to the full set. Expect to edit this document after the pilot — that is the point of the pilot.

---

## 1. The three labels

**`cognate`** — the two words have a **common origin** *and* a **shared/overlapping core meaning**, such that a Telugu speaker seeing the Kannada word (or vice versa) would plausibly recognise it. "Common origin" here is defined broadly (see §2): it includes both words inherited from Proto-Dravidian *and* words both languages borrowed from a shared external source (chiefly Sanskrit). Minor sound differences are expected and fine.

**`false_friend`** — the two words are **similar in form** but their meanings **differ** in a way that would mislead a learner. The form similarity is what makes it a trap; the meaning divergence is what makes it "false." Sub-types to note in the `notes` column (all still labelled `false_friend`):
- *Full false friend* — meanings are unrelated (same-looking word, completely different sense).
- *Partial false friend* — meanings overlap in some contexts but diverge in others, or one has narrowed/broadened/shifted (very common with shared Sanskrit words that drifted in one language).

**`unrelated`** — neither related in origin nor confusably similar in form. These are the negatives; most random pairs fall here.

> Rule of thumb: **origin + meaning → `cognate`; form-similarity + meaning-divergence → `false_friend`; neither → `unrelated`.** A pair that is form-similar *and* meaning-shared is a cognate, not a false friend.

---

## 2. The Kannada–Telugu-specific decision: shared Sanskrit borrowings

This is the most important call for **this** language pair, and it differs from a Kannada–Tamil project.

Kannada and Telugu both absorbed a very large layer of Sanskrit vocabulary (*tatsama* = borrowed largely unchanged; *tadbhava* = borrowed and nativised). Many of your IndoWordNet candidate pairs will therefore be **shared Sanskrit borrowings**, not words inherited from a common Dravidian ancestor. Strictly, in historical linguistics, independently borrowed loanwords are **not** cognates. But for an intercomprehension / learnability purpose, a shared Sanskrit word *is* a shared, learnable word.

**Decision for this project:** label shared-origin **and** shared-meaning pairs as `cognate` **regardless of whether the shared origin is Dravidian inheritance or Sanskrit borrowing.** This matches the project's learnability goal and matches prior work on Indian-language cognate datasets (which folds loanwords into "cognate").

**But** capture the distinction in a separate column so you can do a sub-analysis without changing the main label:

`origin` ∈ { `inherited` (native Dravidian), `sanskrit` (shared tatsama/tadbhava), `other_borrowing` (English, Persian, Portuguese, etc.), `uncertain` }

You do **not** need deep etymological expertise for `origin` — a best-effort guess is fine, and `uncertain` is a valid answer. This column powers an "inherited vs. borrowed" breakdown in the error analysis (a nice result: does the classifier behave differently on inherited cognates vs. shared borrowings?) but never affects the primary three-way label.

---

## 3. Decision procedure (apply in order)

1. **Read both glosses.** Use the English gloss / synset definition to fix each word's meaning. If a word is polysemous, judge against the sense given by the candidate's source synset.
2. **Meaning check.** Do the core meanings match?
   - **Yes** → candidate for `cognate`. Confirm the forms are actually relatable (they came from a shared synset, so usually yes). Set `origin` best-effort. Label `cognate`.
   - **No / only superficial overlap** → go to step 3.
3. **Form check.** Are the (transliterated) forms similar enough that a learner could confuse them?
   - **Yes** → `false_friend` (note full vs. partial).
   - **No** → `unrelated`.

---

## 4. Edge cases and conventions

- **Inflected / derived forms.** Judge on the lemma/stem. If a candidate pair differs only by an inflectional ending, treat the shared stem as the basis for the decision; note it.
- **Transliteration mismatches.** If the ISO-15919 forms disagree only in ways that reflect predictable Kannada↔Telugu orthographic conventions (e.g. inherent-vowel handling, anusvara/nasal spelling, gemination), do **not** count that against a `cognate` judgement — it's the same word.
- **English / other modern borrowings** (e.g. both languages using a form of "bus", "computer"). These *are* shared and learnable → `cognate` with `origin = other_borrowing`. Flag them; you may want to report metrics with and without this class.
- **Proper nouns and named entities.** Exclude unless your team explicitly decides to keep them; note the decision here.
- **Multi-word expressions.** Prefer single-word candidates for the core dataset. If a multi-word pair slips in, label on the head word and flag it.
- **One word, two plausible labels.** When you hesitate between `cognate` and `false_friend`, ask: *would the shared appearance help the learner (→ cognate) or trip them (→ false friend)?* Record the reasoning in `notes` — these are the pairs worth discussing at adjudication.
- **Uncertain.** If you cannot decide even after discussion, mark `label = uncertain` during the pilot. High `uncertain` counts on a category are a signal to sharpen these guidelines before the full run. `uncertain` items are excluded from the final gold set unless resolved.

---

## 5. Where candidates come from (so you know what you're looking at)

Your candidates arrive from **two streams**, because a gold set drawn only from shared synsets would contain almost no false friends:

- **Stream A — cognate candidates:** pairs whose Kannada and Telugu words occupy the **same IndoWordNet synset** (same concept). These are mostly `cognate` / `unrelated`.
- **Stream B — false-friend candidates:** pairs that are **orthographically similar after transliteration but do *not* share a synset** (i.e. similar form, different concept). This is where `false_friend` labels come from. Confirm the meanings genuinely differ before labelling.

The `candidate_source` column records which stream a pair came from. Do not let the source bias your label — a Stream B pair can still turn out to be a `cognate` (real cognate the wordnets happened not to link), and a Stream A pair can be `unrelated`.

---

## 6. Dataset schema (CSV)

One row per candidate pair:

| column | meaning |
|---|---|
| `pair_id` | stable unique id |
| `kn_word` | Kannada word (native script) |
| `te_word` | Telugu word (native script) |
| `kn_iso` | Kannada word in ISO-15919 |
| `te_iso` | Telugu word in ISO-15919 |
| `synset_id` | IndoWordNet synset id (Stream A) or blank (Stream B / random) |
| `gloss` | bilingual native-script gloss: `kn: <Kannada gloss> || te: <Telugu gloss>` |
| `candidate_source` | `shared_synset` (A), `form_similar` (B), or `random` (pilot negatives) |
| `label` | `cognate` / `false_friend` / `unrelated` / `uncertain` |
| `origin` | `inherited` / `sanskrit` / `other_borrowing` / `uncertain` (only meaningful for `cognate`) |
| `annotator` | annotator id |
| `notes` | free text: sub-type, reasoning, flags |

---

## 7. Process

1. Each annotator labels **independently** — no peeking at others' labels.
2. All annotators label a **shared overlap set** (e.g. the same 50–100 pairs) so you can compute **Cohen's kappa** (2 annotators) or **Fleiss' kappa** (3). Report it in the datasheet.
3. **Adjudicate** disagreements in a meeting; update these guidelines with any new rule the disagreement revealed; re-version this file.
4. Split the remaining pairs across annotators for the full run.
5. Keep the per-annotator files *and* the merged gold file. Reproducibility depends on it.
