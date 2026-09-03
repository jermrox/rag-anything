"""Model persistence and the provenance guarantees attached to it."""

import json
from pathlib import Path

import pytest

from vitalgraph.ml.features import FEATURE_NAMES
from vitalgraph.ml.registry import (
    REAL,
    SYNTHETIC,
    FeatureMismatch,
    ModelCard,
    ModelNotFound,
    ModelRegistry,
)


@pytest.fixture()
def registry(tmp_path):
    return ModelRegistry(tmp_path / "models")


def _card(**kw):
    defaults = dict(
        name="demo", algorithm="IsolationForest", training_data=SYNTHETIC, seed=7
    )
    defaults.update(kw)
    return ModelCard(**defaults)


def test_roundtrip_preserves_model_and_card(registry):
    registry.save({"weights": [1, 2, 3]}, _card(n_training_samples=42))
    model, card = registry.load("demo")
    assert model == {"weights": [1, 2, 3]}
    assert card.seed == 7
    assert card.n_training_samples == 42


def test_training_provenance_is_validated():
    """An unrecognised provenance value must fail loudly, not be stored."""
    with pytest.raises(ValueError):
        _card(training_data="probably fine")


def test_synthetic_models_carry_an_explicit_caveat():
    card = _card(training_data=SYNTHETIC)
    assert card.is_synthetic
    assert "simulator" in card.caveat().lower()
    assert "not human physiology" in card.caveat().lower()


def test_real_models_do_not_carry_the_synthetic_caveat():
    card = _card(training_data=REAL, n_training_samples=900)
    assert not card.is_synthetic
    assert "simulator" not in card.caveat().lower()


def test_feature_vocabulary_is_recorded_by_default():
    assert _card().feature_names == FEATURE_NAMES


def test_model_trained_on_other_features_is_refused(registry):
    """Using it anyway would produce confident nonsense."""
    registry.save({"m": 1}, _card())
    path = Path(registry.root) / "demo.json"
    raw = json.loads(path.read_text())
    raw["feature_names"] = ["only", "two"]
    path.write_text(json.dumps(raw))

    with pytest.raises(FeatureMismatch):
        registry.load("demo")
    # Inspection is still possible, deliberately.
    _, card = registry.load("demo", strict_features=False)
    assert list(card.feature_names) == ["only", "two"]


def test_missing_model_raises(registry):
    with pytest.raises(ModelNotFound):
        registry.load("nope")
    with pytest.raises(ModelNotFound):
        registry.card("nope")


def test_listing_surfaces_provenance_without_loading_pickles(registry):
    registry.save({"m": 1}, _card(name="a"))
    registry.save({"m": 2}, _card(name="b", training_data=REAL))
    rows = {r["name"]: r for r in registry.list_models()}
    assert rows["a"]["training_data"] == SYNTHETIC
    assert "simulator" in rows["a"]["caveat"].lower()
    assert rows["b"]["training_data"] == REAL


def test_delete_is_idempotent(registry):
    registry.save({"m": 1}, _card())
    assert registry.delete("demo") is True
    assert registry.delete("demo") is False


def test_trained_at_is_stamped_automatically():
    assert _card().trained_at
    assert _card(trained_at="2026-01-01T00:00:00+00:00").trained_at.startswith(
        "2026-01-01"
    )


def test_qualified_name_includes_version():
    assert _card(version="3").qualified_name == "demo-v3"
