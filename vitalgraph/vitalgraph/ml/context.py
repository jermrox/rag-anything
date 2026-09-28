"""Temporal context: giving each epoch the shape of the hours around it.

Sleep stages are strongly autocorrelated. An epoch surrounded by deep sleep is
very likely deep sleep, and the instantaneous statistics of thirty seconds of
RR intervals are a noisy, weak view of that. Our features carried one epoch of
context either side; the staging literature carries far more.

The method here follows what YASA does (BSD-3-Clause, in the harvested corpus,
`yasa@dfddbd5:src/yasa/staging.py#L339-L347`): a centred triangular-weighted
rolling average over fifteen epochs -- 7.5 minutes -- plus a trailing average
over four epochs, two minutes. The two windows answer different questions. The
centred one asks what part of the night this is; the trailing one asks what
just happened, which is what distinguishes a stage transition from a stable
period.

Triangular weighting rather than flat: a flat window says an epoch seven
positions away matters exactly as much as the one next door, which is false
for a signal this autocorrelated.

**Two honest caveats.**

The centred window uses future epochs, so this is offline scoring of a
completed night, not real-time staging. That is the normal setting for a sleep
report generated in the morning, and it is what the published work does, but a
live "you are now in deep sleep" readout could not use these features and must
not claim their accuracy.

Smoothing is computed strictly within a night. Rolling across a night boundary
would blend one person's physiology into another's -- a leak that would raise
accuracy while destroying the meaning of the number.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Sequence, Tuple

from .epochs import EPOCH_FEATURE_NAMES, EpochSample

#: Centred window, in epochs. Fifteen 30-second epochs is 7.5 minutes.
CENTRED_WINDOW_EPOCHS = 15

#: Trailing window, in epochs. Four is two minutes.
TRAILING_WINDOW_EPOCHS = 4

#: Suffixes for the derived feature names, in the order they are appended.
CENTRED_SUFFIX = "_centred"
TRAILING_SUFFIX = "_trailing"

#: Feature names after augmentation: the originals, then each original
#: smoothed two ways. Kept as a module constant so a model card records what
#: the model was actually trained on rather than what it was assumed to be.
CONTEXT_FEATURE_NAMES: Tuple[str, ...] = (
    EPOCH_FEATURE_NAMES
    + tuple(f"{name}{CENTRED_SUFFIX}" for name in EPOCH_FEATURE_NAMES)
    + tuple(f"{name}{TRAILING_SUFFIX}" for name in EPOCH_FEATURE_NAMES)
)


def triangular_weights(size: int) -> List[float]:
    """Symmetric triangular weights, peaking at the centre.

    Normalised to sum to one so a smoothed feature stays on the same scale as
    the raw one. Without that, every smoothed column would be inflated by the
    window size and the model would see a difference of magnitude where there
    is only a difference of smoothing.
    """
    if size <= 0:
        return []
    if size == 1:
        return [1.0]
    half = size // 2
    raw: List[float] = []
    for i in range(size):
        distance = abs(i - half)
        raw.append(max(0.0, half + 1 - distance))
    total = sum(raw)
    return [w / total for w in raw] if total else [1.0 / size] * size


def _centred_average(values: Sequence[float], index: int, window: int) -> float:
    """Triangular-weighted average around ``index``, clipped at the edges.

    Weights are renormalised over whatever part of the window exists, so the
    first and last epochs of a night are smoothed with a half window rather
    than being padded with zeros. Padding would drag both ends of every night
    toward zero and invent a transition that did not happen.
    """
    half = window // 2
    low = max(0, index - half)
    high = min(len(values), index + half + 1)
    weights = triangular_weights(window)
    used = weights[low - (index - half) : window - ((index + half + 1) - high)]
    total = sum(used)
    if not used or total == 0:
        return values[index]
    return sum(v * w for v, w in zip(values[low:high], used)) / total


def _trailing_average(values: Sequence[float], index: int, window: int) -> float:
    """Mean of this epoch and the ones just before it."""
    low = max(0, index - window + 1)
    chunk = values[low : index + 1]
    return sum(chunk) / len(chunk) if chunk else values[index]


def add_temporal_context(
    samples: Sequence[EpochSample],
    centred_window: int = CENTRED_WINDOW_EPOCHS,
    trailing_window: int = TRAILING_WINDOW_EPOCHS,
) -> List[EpochSample]:
    """Append smoothed versions of every feature, computed within each night.

    Returns samples in their original order, so a caller's pairing with labels
    or night ids is unaffected.
    """
    by_night: Dict[str, List[int]] = {}
    for position, sample in enumerate(samples):
        by_night.setdefault(sample.night_id, []).append(position)

    augmented: List[EpochSample | None] = [None] * len(samples)

    for positions in by_night.values():
        # Order by epoch index, not by list order: a caller may hand us
        # samples grouped or shuffled, and smoothing the wrong sequence would
        # produce plausible-looking nonsense rather than an error.
        ordered = sorted(positions, key=lambda p: samples[p].index)
        n_features = len(samples[ordered[0]].values)
        columns = [[samples[p].values[f] for p in ordered] for f in range(n_features)]

        for slot, position in enumerate(ordered):
            centred = [
                _centred_average(columns[f], slot, centred_window)
                for f in range(n_features)
            ]
            trailing = [
                _trailing_average(columns[f], slot, trailing_window)
                for f in range(n_features)
            ]
            augmented[position] = replace(
                samples[position],
                values=tuple(samples[position].values)
                + tuple(centred)
                + tuple(trailing),
            )

    return [s for s in augmented if s is not None]
