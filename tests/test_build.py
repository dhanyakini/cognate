"""Tests for cognate.build — imputation flag (synthetic only)."""

from __future__ import annotations

from cognate.build import SEMANTIC_IMPUTE_VALUE, drop_uncertain, featurize_row


def test_semantic_imputation_flag(monkeypatch) -> None:
    row = {
        "pair_id": "X1",
        "kn_iso": "abc",
        "te_iso": "abc",
        "en_kn": "",
        "en_te": "",
        "final_label": "cognate",
    }

    monkeypatch.setattr(
        "cognate.build.normalized_similarity", lambda a, b: 1.0
    )
    monkeypatch.setattr(
        "cognate.build.phonetic_similarity", lambda a, b: 0.9
    )
    monkeypatch.setattr(
        "cognate.build.semantic_similarity", lambda *a, **k: None
    )

    out = featurize_row(row)
    assert out["semantic_imputed"] == "True"
    assert float(out["sem_sim"]) == SEMANTIC_IMPUTE_VALUE
    assert out["label"] == "cognate"


def test_semantic_not_imputed_when_present(monkeypatch) -> None:
    row = {
        "pair_id": "X1",
        "kn_iso": "abc",
        "te_iso": "xyz",
        "en_kn": "water",
        "en_te": "water",
        "final_label": "cognate",
    }
    monkeypatch.setattr(
        "cognate.build.normalized_similarity", lambda a, b: 0.2
    )
    monkeypatch.setattr(
        "cognate.build.phonetic_similarity", lambda a, b: 0.3
    )
    monkeypatch.setattr(
        "cognate.build.semantic_similarity", lambda *a, **k: 0.91
    )
    out = featurize_row(row)
    assert out["semantic_imputed"] == "False"
    assert float(out["sem_sim"]) == 0.91


def test_drop_uncertain_warns(capsys) -> None:
    rows = [
        {"pair_id": "A", "final_label": "cognate"},
        {"pair_id": "B", "final_label": "uncertain"},
        {"pair_id": "C", "final_label": "unrelated"},
    ]
    kept = drop_uncertain(rows)
    assert [r["pair_id"] for r in kept] == ["A", "C"]
    err = capsys.readouterr().err
    assert "dropped 1" in err
    assert "uncertain" in err
