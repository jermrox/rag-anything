"""Tests for temporal context features.

The load-bearing property is that smoothing never crosses a night boundary.
Rolling one person's physiology into another's would raise accuracy while
destroying the meaning of the number, which is the worst kind of bug this
project can have: one that looks like progress.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from vitalgraph.ml.context import (
    CENTRED_SUFFIX,
    CENTRED_WINDOW_EPOCHS,
    CONTEXT_FEATURE_NAMES,
    TRAILING_SUFFIX,
    TRAILING_WINDOW_EPOCHS,
    add_temporal_context,
    triangular_weights,
)
from vitalgraph.ml.epochs import EPOCH_FEATURE_NAMES, EpochSample

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
N_FEATURES = len(EPOCH_FEATURE_NAMES)


def _epoch(night: str, index: int, value: float) -> EpochSample:
    return EpochSample(
        night_id=night,
        index=index,
        start=START + timedelta(seconds=30 * index),
        values=(value,) * N_FEATURES,
        label=1.0,
    )


# --- weights ---------------------------------------------------------------


def test_triangular_weights_sum_to_one():
    """Smoothed features must stay on the raw feature's scale.

    Unnormalised weights inflate every smoothed column by the window size, and
    the model sees a difference of magnitude where there is only a difference
    of smoothing.
    """
    for size in (1, 2, 5, 15):
        assert sum(triangular_weights(size)) == pytest.approx(1.0)


def test_triangular_weights_peak_at_the_centre():
    """A flat window would say an epoch seven positions away matters as much
    as the one next door, which is false for a signal this autocorrelated."""
    weights = triangular_weights(15)
    assert weights[7] == max(weights)
    assert weights[0] < weights[7]
    assert weights[0] == pytest.approx(weights[-1])


def test_degenerate_window_sizes_are_handled():
    assert triangular_weights(0) == []
    assert triangular_weights(1) == [1.0]


# --- the shape of the output ----------------------------------------------


def test_context_triples_the_feature_count():
    samples = [_epoch("n1", i, float(i)) for i in range(20)]
    augmented = add_temporal_context(samples)
    assert len(augmented[0].values) == N_FEATURES * 3
    assert len(CONTEXT_FEATURE_NAMES) == N_FEATURES * 3


def test_the_original_features_are_preserved_unchanged():
    """The raw values stay first, so a model trained on the augmented set can
    still be compared with one trained on the originals."""
    samples = [_epoch("n1", i, float(i)) for i in range(20)]
    augmented = add_temporal_context(samples)
    for original, result in zip(samples, augmented):
        assert result.values[:N_FEATURES] == original.values


def test_feature_names_line_up_with_the_values():
    assert CONTEXT_FEATURE_NAMES[:N_FEATURES] == EPOCH_FEATURE_NAMES
    assert CONTEXT_FEATURE_NAMES[N_FEATURES].endswith(CENTRED_SUFFIX)
    assert CONTEXT_FEATURE_NAMES[2 * N_FEATURES].endswith(TRAILING_SUFFIX)


def test_order_and_identity_are_preserved():
    """A caller's pairing with labels or night ids must survive."""
    samples = [_epoch("n1", i, float(i)) for i in range(10)]
    augmented = add_temporal_context(samples)
    assert [s.index for s in augmented] == [s.index for s in samples]
    assert [s.label for s in augmented] == [s.label for s in samples]
    assert [s.night_id for s in augmented] == [s.night_id for s in samples]


# --- the leak that matters -------------------------------------------------


def test_smoothing_never_crosses_a_night_boundary():
    """The load-bearing property.

    Two nights with wildly different levels. If smoothing rolled across the
    boundary, the last epochs of the first night would be pulled toward the
    second night's level -- one person's physiology leaking into another's.
    """
    first = [_epoch("n1", i, 0.0) for i in range(20)]
    second = [_epoch("n2", i, 100.0) for i in range(20)]
    augmented = add_temporal_context(first + second)

    from_first = [s for s in augmented if s.night_id == "n1"]
    from_second = [s for s in augmented if s.night_id == "n2"]

    # Every smoothed value stays at its own night's level.
    for sample in from_first:
        assert all(v == pytest.approx(0.0) for v in sample.values[N_FEATURES:])
    for sample in from_second:
        assert all(v == pytest.approx(100.0) for v in sample.values[N_FEATURES:])


def test_epochs_are_smoothed_in_index_order_not_list_order():
    """A caller may hand us samples shuffled or grouped.

    Smoothing the wrong sequence produces plausible-looking nonsense rather
    than an error, so ordering is by epoch index explicitly.
    """
    ordered = [_epoch("n1", i, float(i)) for i in range(20)]
    shuffled = [
        ordered[i]
        for i in (5, 0, 12, 3, 19, 7, 1, 2, 4, 6, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18)
    ]

    from_ordered = {s.index: s.values for s in add_temporal_context(ordered)}
    from_shuffled = {s.index: s.values for s in add_temporal_context(shuffled)}
    assert from_ordered == from_shuffled


# --- smoothing behaviour ---------------------------------------------------


def test_a_constant_night_smooths_to_the_same_constant():
    """Nothing is invented where nothing varies."""
    samples = [_epoch("n1", i, 7.0) for i in range(30)]
    augmented = add_temporal_context(samples)
    for sample in augmented:
        assert all(v == pytest.approx(7.0) for v in sample.values)


def test_edges_are_smoothed_with_a_half_window_not_padded_with_zeros():
    """Padding would drag both ends of every night toward zero and invent a
    transition that did not happen."""
    samples = [_epoch("n1", i, 5.0) for i in range(30)]
    augmented = add_temporal_context(samples)
    first, last = augmented[0], augmented[-1]
    assert first.values[N_FEATURES] == pytest.approx(5.0)
    assert last.values[N_FEATURES] == pytest.approx(5.0)


def test_centred_smoothing_reduces_the_amplitude_of_a_spike():
    """That is what smoothing is for: one loud epoch should not dominate."""
    samples = [_epoch("n1", i, 0.0) for i in range(30)]
    samples[15] = _epoch("n1", 15, 100.0)
    augmented = add_temporal_context(samples)
    spike = next(s for s in augmented if s.index == 15)
    assert spike.values[0] == pytest.approx(100.0)  # raw untouched
    assert spike.values[N_FEATURES] < 50.0  # centred, much reduced


def test_the_trailing_window_looks_backwards_only():
    """A step change must not appear in the trailing feature before it happens.

    The centred window legitimately sees the future; the trailing one must
    not, since it is what distinguishes a transition from a stable period.
    """
    samples = [_epoch("n1", i, 0.0 if i < 15 else 100.0) for i in range(30)]
    augmented = add_temporal_context(samples)
    trailing_index = 2 * N_FEATURES

    before = next(s for s in augmented if s.index == 14)
    assert before.values[trailing_index] == pytest.approx(0.0)

    at_step = next(s for s in augmented if s.index == 15)
    assert at_step.values[trailing_index] > 0.0


def test_window_constants_are_the_published_ones():
    """15 epochs is 7.5 minutes and 4 is 2 minutes, per the method followed."""
    assert CENTRED_WINDOW_EPOCHS * 30 / 60 == pytest.approx(7.5)
    assert TRAILING_WINDOW_EPOCHS * 30 / 60 == pytest.approx(2.0)


def test_a_single_epoch_night_is_returned_unharmed():
    augmented = add_temporal_context([_epoch("n1", 0, 3.0)])
    assert len(augmented) == 1
    assert all(v == pytest.approx(3.0) for v in augmented[0].values)


def test_empty_input_yields_empty_output():
    assert add_temporal_context([]) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
