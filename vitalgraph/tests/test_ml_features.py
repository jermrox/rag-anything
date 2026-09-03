"""Feature extraction. Pure standard library, so it runs without the ML extra."""

import math

import pytest

from vitalgraph.biometrics.schema import utc
from vitalgraph.biometrics.store import BiometricStore
from vitalgraph.ble.simulator import simulate_night, simulate_period
from vitalgraph.bridge.summarizer import nightly_windows
from vitalgraph.ml.features import (
    FEATURE_NAMES,
    FeatureVector,
    features_for_windows,
    night_features,
    to_matrix,
)

START = utc(1772582400)  # 2026-03-04T00:00:00Z


@pytest.fixture()
def store():
    s = BiometricStore(":memory:")
    s.add(
        simulate_period(
            START, nights=4, recovery_by_night=[1.0, 0.95, 0.25, 0.2], seed=11
        )
    )
    yield s
    s.close()


def test_vector_length_matches_the_vocabulary(store):
    fv = night_features(store, START, utc(START.timestamp() + 8 * 3600), "n1")
    assert fv is not None
    assert len(fv.values) == len(FEATURE_NAMES)


def test_feature_vocabulary_has_no_duplicates():
    """A duplicated name would make `get()` ambiguous and silently wrong."""
    assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)


def test_wrong_length_is_rejected():
    with pytest.raises(ValueError):
        FeatureVector(period_id="x", start=START, end=START, values=(1.0, 2.0))


def test_insufficient_data_yields_none_not_zeros(store):
    """A model cannot distinguish 'measured zero' from 'no data', so an
    under-populated window must produce no vector at all."""
    empty = BiometricStore(":memory:")
    assert night_features(empty, START, utc(START.timestamp() + 3600), "empty") is None


def test_features_discriminate_recovery(store):
    vectors = features_for_windows(store, nightly_windows(START, 4))
    good, bad = vectors[0].as_dict(), vectors[-1].as_dict()
    assert bad["rmssd_ms"] < good["rmssd_ms"]
    assert bad["mean_hr_bpm"] > good["mean_hr_bpm"]
    assert bad["awake_fraction"] > good["awake_fraction"]
    assert bad["fragmentation"] > good["fragmentation"]


def test_sleep_fractions_are_bounded(store):
    for fv in features_for_windows(store, nightly_windows(START, 4)):
        d = fv.as_dict()
        for key in (
            "deep_fraction",
            "rem_fraction",
            "light_fraction",
            "awake_fraction",
        ):
            assert 0.0 <= d[key] <= 1.0
        stages = d["deep_fraction"] + d["rem_fraction"] + d["light_fraction"]
        assert stages == pytest.approx(1.0, abs=1e-6)


def test_circadian_encoding_is_cyclic():
    """23:00 and 01:00 are two hours apart, not twenty-two."""
    s = BiometricStore(":memory:")
    late = utc(START.timestamp() + 23 * 3600)
    early = utc(START.timestamp() + 25 * 3600)
    s.add(simulate_night(late, hours=8, seed=1))
    s.add(simulate_night(early, hours=8, seed=1))

    a = night_features(s, late, utc(late.timestamp() + 8 * 3600), "late")
    b = night_features(s, early, utc(early.timestamp() + 8 * 3600), "early")
    distance = math.dist(
        (a.get("start_hour_sin"), a.get("start_hour_cos")),
        (b.get("start_hour_sin"), b.get("start_hour_cos")),
    )
    assert distance < 0.6  # far smaller than the diameter of the unit circle


def test_extraction_is_deterministic(store):
    windows = nightly_windows(START, 4)
    assert to_matrix(features_for_windows(store, windows)) == to_matrix(
        features_for_windows(store, windows)
    )


def test_windows_without_data_are_skipped_not_faked(store):
    vectors = features_for_windows(store, nightly_windows(START, 30))
    assert len(vectors) == 4  # only the nights that exist


def test_get_by_name_matches_dict(store):
    fv = features_for_windows(store, nightly_windows(START, 1))[0]
    assert fv.get("rmssd_ms") == fv.as_dict()["rmssd_ms"]
