"""Deterministic synthetic BLE health stream.

No hardware exists yet, so the simulator is the backbone of development *and*
of the test suite: seeded, so a given seed always yields identical beats, and
physiologically structured, so metrics computed from it move in the directions
real physiology moves.

Modelled effects:

* ~90 minute sleep cycles through Light / Deep / REM,
* heart rate dipping in deep sleep and rising toward morning,
* respiratory sinus arrhythmia -- RR oscillating with the breathing cycle,
  which is what makes RMSSD non-zero and what a respiratory-rate estimator
  later recovers,
* a ``recovery`` dial that jointly suppresses HRV, raises resting HR, raises
  skin temperature and fragments sleep, exactly as poor recovery presents.
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import List

from ..biometrics.schema import Sample, SignalType, SleepStage

SOURCE = "simulator"

# A full sleep cycle: light -> deep -> light -> REM.
CYCLE_MINUTES = 90.0


#: Sleep is scored in 30-second epochs by convention; stages persist across an
#: epoch rather than being redrawn per beat (redrawing per beat would make
#: heart rate jump every second, which is not physiology).
EPOCH_SECONDS = 30.0


def _stage_track(
    total_seconds: float, rng: random.Random, fragmentation: float
) -> List[SleepStage]:
    """Pre-compute the whole night's hypnogram, one entry per 30 s epoch.

    Deep sleep dominates early cycles and REM dominates later ones, matching
    normal sleep architecture. ``fragmentation`` injects brief awakenings, which
    persist for a full epoch instead of flickering beat to beat.
    """
    n_epochs = int(total_seconds / EPOCH_SECONDS) + 1
    track: List[SleepStage] = []
    for epoch in range(n_epochs):
        minutes_in = epoch * EPOCH_SECONDS / 60.0
        if rng.random() < fragmentation:
            track.append(SleepStage.AWAKE)
            continue

        phase = (minutes_in % CYCLE_MINUTES) / CYCLE_MINUTES
        night_progress = min(minutes_in / 480.0, 1.0)  # 8h night

        # Deep-sleep pressure decays across the night; REM pressure grows.
        deep_window = 0.55 * (1.0 - night_progress)
        rem_window = 0.30 * night_progress

        if phase < deep_window:
            track.append(SleepStage.DEEP)
        elif phase > 1.0 - rem_window:
            track.append(SleepStage.REM)
        else:
            track.append(SleepStage.LIGHT)
    return track


def simulate_night(
    start: datetime,
    hours: float = 8.0,
    recovery: float = 1.0,
    seed: int = 0,
) -> List[Sample]:
    """Generate one night of biometric samples.

    Args:
        start: timezone-aware start of the sleep period.
        hours: sleep duration.
        recovery: 1.0 = well recovered, 0.0 = badly recovered. Drives HRV
            suppression, resting-HR elevation, temperature and fragmentation.
        seed: RNG seed; identical seeds produce byte-identical output.

    Returns:
        Samples ordered by time, mixing RR intervals, heart rate, SpO2,
        skin temperature and sleep stage.
    """
    if start.tzinfo is None:
        raise ValueError("start must be timezone-aware")
    recovery = max(0.0, min(1.0, recovery))
    rng = random.Random(seed)

    # Poor recovery: higher resting HR, blunted RSA, warmer skin, broken sleep.
    resting_hr = 62.0 - 8.0 * recovery + 10.0 * (1.0 - recovery)
    rsa_amplitude = 12.0 + 38.0 * recovery  # ms of RR swing with breathing
    beat_noise = 4.0 + 6.0 * (1.0 - recovery)  # ms of stochastic variation
    base_temp = 33.4 + 0.9 * (1.0 - recovery)
    fragmentation = 0.01 + 0.09 * (1.0 - recovery)
    respiration_hz = 0.22 + 0.04 * (1.0 - recovery)  # ~13-16 breaths/min

    samples: List[Sample] = []
    elapsed = 0.0  # seconds into the night
    total_seconds = hours * 3600.0
    last_minute_logged = -1
    stage_track = _stage_track(total_seconds, rng, fragmentation)

    while elapsed < total_seconds:
        minutes_in = elapsed / 60.0
        stage = stage_track[int(elapsed / EPOCH_SECONDS)]

        # Deep sleep lowers HR; waking raises it. Slight morning rise.
        stage_offset = {
            SleepStage.DEEP: -6.0,
            SleepStage.LIGHT: 0.0,
            SleepStage.REM: 3.0,
            SleepStage.AWAKE: 9.0,
        }[stage]
        morning_rise = 3.0 * (minutes_in / (hours * 60.0))
        target_hr = resting_hr + stage_offset + morning_rise

        base_rr = 60000.0 / target_hr

        # Respiratory sinus arrhythmia: RR rides the breathing cycle. Deep
        # sleep shows the strongest vagal modulation.
        stage_gain = {
            SleepStage.DEEP: 1.30,
            SleepStage.LIGHT: 1.0,
            SleepStage.REM: 0.70,
            SleepStage.AWAKE: 0.45,
        }[stage]
        rsa = (
            rsa_amplitude
            * stage_gain
            * math.sin(2 * math.pi * respiration_hz * elapsed)
        )
        rr = base_rr + rsa + rng.gauss(0.0, beat_noise)
        rr = max(300.0, min(2000.0, rr))

        ts = start + timedelta(seconds=elapsed)
        samples.append(
            Sample(ts=ts, signal=SignalType.RR_INTERVAL, value=rr, source=SOURCE)
        )

        # Lower-rate channels, once per minute.
        minute = int(minutes_in)
        if minute != last_minute_logged:
            last_minute_logged = minute
            samples.append(
                Sample(
                    ts=ts,
                    signal=SignalType.HEART_RATE,
                    value=round(60000.0 / rr, 1),
                    source=SOURCE,
                )
            )
            samples.append(
                Sample(
                    ts=ts,
                    signal=SignalType.SLEEP_STAGE,
                    value=float(stage.value),
                    source=SOURCE,
                )
            )
            samples.append(
                Sample(
                    ts=ts,
                    signal=SignalType.SPO2,
                    value=round(
                        min(100.0, 97.5 - 1.5 * (1 - recovery) + rng.gauss(0, 0.4)), 1
                    ),
                    source=SOURCE,
                )
            )
            # Skin temperature follows a shallow nocturnal curve.
            temp_curve = 0.4 * math.sin(math.pi * minutes_in / (hours * 60.0))
            samples.append(
                Sample(
                    ts=ts,
                    signal=SignalType.SKIN_TEMPERATURE,
                    value=round(base_temp + temp_curve + rng.gauss(0, 0.05), 2),
                    source=SOURCE,
                )
            )

        elapsed += rr / 1000.0  # advance by one beat

    return samples


def simulate_period(
    start: datetime,
    nights: int = 7,
    recovery_by_night: List[float] | None = None,
    seed: int = 0,
) -> List[Sample]:
    """Generate consecutive nights, one per 24h, for baseline-vs-episode tests.

    ``recovery_by_night`` lets a caller script a narrative -- e.g. five good
    nights then two bad ones -- which is exactly the shape the episode
    summariser needs to produce an interesting answer.
    """
    if recovery_by_night is None:
        recovery_by_night = [1.0] * nights
    if len(recovery_by_night) != nights:
        raise ValueError("recovery_by_night must have one entry per night")

    out: List[Sample] = []
    for i, recovery in enumerate(recovery_by_night):
        out.extend(
            simulate_night(start + timedelta(days=i), recovery=recovery, seed=seed + i)
        )
    return out
