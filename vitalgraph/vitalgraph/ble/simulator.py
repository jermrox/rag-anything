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
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

from ..biometrics.schema import Sample, SignalType, SleepStage

SOURCE = "simulator"

#: SpO2 sampling interval, seconds. Sampling once per minute cannot detect a
#: 45-second desaturation at all -- the event falls between samples. Real pulse
#: oximeters stream continuously for exactly this reason, so the generator has
#: to as well or event detection is untestable by construction.
SPO2_INTERVAL_S = 5.0

#: Duration of an injected apnea-like episode, seconds. Real obstructive events
#: last 10-60 s; the desaturation trails the airway event by roughly this long
#: again, which is why the SpO2 nadir lags the heart-rate response.
APNEA_DURATION_S = 45.0

#: Points of SpO2 lost at the nadir of an injected episode. A drop of 3-4
#: points is the conventional scoring threshold for a desaturation event.
APNEA_DESAT_POINTS = 6.0

#: Duration of an injected irregularity episode, seconds.
ARRHYTHMIA_DURATION_S = 60.0

#: Fraction by which RR intervals scatter during an irregularity episode. Large
#: enough to be unmistakable, and large enough that artifact correction will
#: reject some of it -- which is itself realistic.
ARRHYTHMIA_SCATTER = 0.28


@dataclass(frozen=True)
class Episode:
    """An injected physiological event, with its ground-truth window."""

    kind: str
    """apnea | arrhythmia"""
    start_s: float
    end_s: float

    def contains(self, elapsed: float) -> bool:
        return self.start_s <= elapsed < self.end_s


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


def _plan_episodes(
    total_seconds: float, apnea_events: int, arrhythmia_events: int, rng: random.Random
) -> List[Episode]:
    """Place episodes at non-overlapping random times through the night."""
    episodes: List[Episode] = []
    wanted = [("apnea", APNEA_DURATION_S)] * apnea_events + [
        ("arrhythmia", ARRHYTHMIA_DURATION_S)
    ] * arrhythmia_events

    for kind, duration in wanted:
        for _ in range(50):  # bounded retry; give up rather than loop forever
            begin = rng.uniform(600.0, max(601.0, total_seconds - duration - 600.0))
            candidate = Episode(kind, begin, begin + duration)
            if not any(
                candidate.start_s < e.end_s and e.start_s < candidate.end_s
                for e in episodes
            ):
                episodes.append(candidate)
                break
    return sorted(episodes, key=lambda e: e.start_s)


def simulate_night(
    start: datetime,
    hours: float = 8.0,
    recovery: float = 1.0,
    seed: int = 0,
    apnea_events: int = 0,
    arrhythmia_events: int = 0,
    return_episodes: bool = False,
) -> List[Sample] | Tuple[List[Sample], List[Episode]]:
    """Generate one night of biometric samples.

    Args:
        start: timezone-aware start of the sleep period.
        hours: sleep duration.
        recovery: 1.0 = well recovered, 0.0 = badly recovered. Drives HRV
            suppression, resting-HR elevation, temperature and fragmentation.
        seed: RNG seed; identical seeds produce byte-identical output.
        apnea_events: number of apnea-like episodes to inject. Each drops SpO2
            by roughly ``APNEA_DESAT_POINTS`` with a compensatory heart-rate
            rise. Without these the SpO2 channel has no events in it at all, so
            a desaturation detector cannot be evaluated against this generator.
        arrhythmia_events: number of irregularity episodes to inject, during
            which RR intervals scatter well beyond normal variability.
        return_episodes: also return the ground-truth episode windows, which is
            what makes event detection measurable rather than merely plausible.

    Returns:
        Samples ordered by time, mixing RR intervals, heart rate, SpO2,
        skin temperature and sleep stage. With ``return_episodes``, a
        ``(samples, episodes)`` pair.
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
    episodes = _plan_episodes(total_seconds, apnea_events, arrhythmia_events, rng)
    last_spo2_bucket = -1

    def active(kind: str, at: float) -> Episode | None:
        for episode in episodes:
            if episode.kind == kind and episode.contains(at):
                return episode
        return None

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

        if active("arrhythmia", elapsed) is not None:
            rr *= 1.0 + rng.gauss(0.0, ARRHYTHMIA_SCATTER)

        if active("apnea", elapsed) is not None:
            # Airway events drive a compensatory tachycardia, which appears
            # before the desaturation reaches the periphery.
            rr *= 0.90

        rr = max(300.0, min(2000.0, rr))

        ts = start + timedelta(seconds=elapsed)
        samples.append(
            Sample(ts=ts, signal=SignalType.RR_INTERVAL, value=rr, source=SOURCE)
        )

        # SpO2 on its own faster cadence, so short events are observable.
        spo2_bucket = int(elapsed / SPO2_INTERVAL_S)
        if spo2_bucket != last_spo2_bucket:
            last_spo2_bucket = spo2_bucket
            spo2 = 97.5 - 1.5 * (1 - recovery) + rng.gauss(0, 0.4)
            apnea = active("apnea", elapsed)
            if apnea is not None:
                # Desaturation follows a rough triangle, deepest mid-episode.
                phase = (elapsed - apnea.start_s) / (apnea.end_s - apnea.start_s)
                spo2 -= APNEA_DESAT_POINTS * (1.0 - abs(2.0 * phase - 1.0))
            samples.append(
                Sample(
                    ts=ts,
                    signal=SignalType.SPO2,
                    value=round(max(50.0, min(100.0, spo2)), 1),
                    source=SOURCE,
                )
            )

        # Remaining low-rate channels, once per minute.
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

    if return_episodes:
        return samples, episodes
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
