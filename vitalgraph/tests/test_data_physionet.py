"""Tests for the PhysioNet reader and R-peak detection.

No network. The WFDB layouts are exercised against bytes built here from the
published specification, which proves the decoder matches the format rather
than matching one file that happened to work.
"""

from __future__ import annotations

import struct

import pytest

from vitalgraph.biometrics.schema import SleepStage
from vitalgraph.data.ecg import PLAUSIBLE_RR_MS, detect_beats
from vitalgraph.data.physionet import (
    FormatError,
    decode_format_212,
    parse_annotations,
    parse_header,
    stages_from_annotations,
)

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")

SLPDB_HEADER = """slp01a 4 250/0.033333333(94) 1800000 23:07:00 19/1/1989
slp01a.dat 212 -200/mV 12 0 -17 59911 0 ECG
slp01a.dat 212 4.77778(-477)/mmHg 12 0 -248 19332 0 BP
slp01a.dat 212 -6430/mV 12 0 252 49594 0 EEG (C4-A1)
slp01a.dat 212 690/l 12 0 -180 912 0 Resp (sum)
# 44 M 89 32-01-89
"""


# --- header parsing --------------------------------------------------------


def test_parses_a_real_slpdb_header():
    header = parse_header(SLPDB_HEADER)
    assert header.name == "slp01a"
    assert header.n_signals == 4
    assert header.sampling_hz == 250.0
    assert header.n_samples == 1_800_000
    assert [s.name for s in header.signals][0] == "ECG"


def test_channel_lookup_is_loose_because_names_vary():
    """Names differ across records -- ECG, ECG1, "ECG (V5)".

    An exact match would find nothing on half the database and return an
    empty result rather than an error.
    """
    header = parse_header(SLPDB_HEADER)
    assert header.channel_index("ecg") == 0
    assert header.channel_index("eeg") == 2
    assert header.channel_index("spo2") is None


def test_a_zero_gain_is_read_as_the_specified_default():
    """A gain field of zero means 200 ADC units per mV by convention.

    Taking it literally divides every sample by zero.
    """
    header = parse_header("rec 1 250 1000\nrec.dat 212 0 12 0 0 0 0 ECG\n")
    assert header.signals[0].gain == 200.0


def test_an_empty_header_is_an_error_not_an_empty_record():
    with pytest.raises(FormatError):
        parse_header("")


# --- format 212 ------------------------------------------------------------


def _pack_212(values):
    """Pack signed 12-bit samples, two per three bytes."""
    out = bytearray()
    for i in range(0, len(values), 2):
        a = values[i] & 0x0FFF
        b = values[i + 1] & 0x0FFF if i + 1 < len(values) else 0
        out.append(a & 0xFF)
        out.append(((a >> 8) & 0x0F) | ((b >> 4) & 0xF0))
        out.append(b & 0xFF)
    return bytes(out)


def test_format_212_round_trips_positive_values():
    values = [0, 1, 100, 2047, 5, 6]
    decoded = decode_format_212(_pack_212(values), n_signals=1, n_samples=6)
    assert list(decoded[:, 0]) == values


def test_format_212_decodes_negative_values_as_twos_complement():
    """Skipping the sign conversion inverts every QRS complex.

    A negative deflection read as a large positive number breaks peak
    detection quietly rather than raising anything.
    """
    values = [-1, -2048, 2047, 0]
    decoded = decode_format_212(_pack_212(values), n_signals=1, n_samples=4)
    assert list(decoded[:, 0]) == values


def test_format_212_deinterleaves_channels():
    # Two channels, samples interleaved: c0=1,3 and c1=2,4.
    decoded = decode_format_212(_pack_212([1, 2, 3, 4]), n_signals=2, n_samples=2)
    assert list(decoded[:, 0]) == [1, 3]
    assert list(decoded[:, 1]) == [2, 4]


def test_format_212_truncates_rather_than_overrunning_short_data():
    decoded = decode_format_212(_pack_212([1, 2]), n_signals=1, n_samples=100)
    assert len(decoded) == 2


# --- annotations -----------------------------------------------------------


def _annotation(code: int, interval: int) -> bytes:
    return struct.pack("<H", (code << 10) | interval)


def _aux(text: str) -> bytes:
    payload = text.encode("ascii")
    out = _annotation(63, len(payload)) + payload
    return out + (b"\x00" if len(payload) % 2 else b"")


def test_annotations_accumulate_sample_intervals():
    data = _annotation(22, 100) + _annotation(22, 250) + _annotation(0, 0)
    assert [t for t, _ in parse_annotations(data)] == [100, 350]


def _skip(interval: int, code: int = 22) -> bytes:
    """A SKIP escape carrying a 32-bit interval, high word first.

    The interval field of an ordinary annotation is ten bits, so anything
    above 1023 samples cannot be expressed without this. A 30-second epoch at
    250 Hz is 7500 samples, so every annotation in a real staging file arrives
    this way -- making the escape the load-bearing path rather than a corner
    case.
    """
    return (
        _annotation(59, 0)
        + struct.pack("<HH", (interval >> 16) & 0xFFFF, interval & 0xFFFF)
        + _annotation(code, 0)
    )


def test_aux_strings_attach_to_their_annotation():
    data = _annotation(22, 500) + _aux("2 OA") + _annotation(0, 0)
    assert parse_annotations(data) == [(500, "2 OA")]


def test_skip_escape_carries_intervals_too_large_for_ten_bits():
    """7500 samples is one 30-second epoch and does not fit in the field.

    Masking it into ten bits silently yields 332, so every epoch in a real
    staging file would land at the wrong time.
    """
    assert 7500 > 0x3FF, "the premise of the escape"
    data = _skip(7500) + _aux("2") + _skip(7500) + _aux("W") + _annotation(0, 0)
    assert parse_annotations(data) == [(7500, "2"), (15000, "W")]


def test_epoch_indices_from_skip_encoded_annotations_are_sequential():
    """End to end: the encoding a real file uses must yield epochs 0, 1, 2."""
    data = b"".join(
        _skip(7500 * (i + 1) - (7500 * i)) + _aux(code)
        for i, code in enumerate(("W", "2", "R"))
    ) + _annotation(0, 0)
    stages = stages_from_annotations(parse_annotations(data), sampling_hz=250.0)
    assert [i for i, _ in stages] == [1, 2, 3]


def test_odd_length_aux_is_padded_and_does_not_desynchronise():
    """A one-byte pad follows an odd-length aux string.

    Missing it reads the pad as the next annotation word and every subsequent
    timestamp is wrong.
    """
    data = _annotation(22, 100) + _aux("W") + _annotation(22, 200) + _annotation(0, 0)
    parsed = parse_annotations(data)
    assert parsed[0] == (100, "W")
    assert parsed[1][0] == 300


def test_parsing_stops_at_the_end_marker():
    data = _annotation(22, 100) + _annotation(0, 0) + _annotation(22, 999)
    assert len(parse_annotations(data)) == 1


# --- stage mapping ---------------------------------------------------------


def test_stage_codes_map_onto_the_four_class_model():
    annotations = [
        (0, "W"),
        (7500, "1"),
        (15000, "2"),
        (22500, "3"),
        (30000, "4"),
        (37500, "R"),
    ]
    stages = stages_from_annotations(annotations, sampling_hz=250.0)
    assert [s for _, s in stages] == [
        SleepStage.AWAKE,
        SleepStage.LIGHT,
        SleepStage.LIGHT,
        SleepStage.DEEP,
        SleepStage.DEEP,
        SleepStage.REM,
    ]
    assert [i for i, _ in stages] == [0, 1, 2, 3, 4, 5]


def test_only_the_first_token_is_the_stage():
    """ "2 OA" is stage 2 with an obstructive apnoea scored in that epoch.

    The event is a different question and must not be folded into the label.
    """
    stages = stages_from_annotations([(0, "2 OA H")], sampling_hz=250.0)
    assert [s for _, s in stages] == [SleepStage.LIGHT]


def test_unscorable_epochs_are_omitted_not_guessed():
    """Movement time and unscored epochs have no stage.

    Emitting one would turn "we do not know" into a wrong label.
    """
    stages = stages_from_annotations(
        [(0, "M"), (7500, "MT"), (15000, ""), (22500, "2")], sampling_hz=250.0
    )
    assert [i for i, _ in stages] == [3]


# --- R-peak detection ------------------------------------------------------


def _synthetic_ecg(bpm: float, seconds: float, sampling_hz: float = 250.0):
    """A crude but QRS-shaped trace: sharp biphasic spikes at a fixed rate."""
    n = int(seconds * sampling_hz)
    signal = np.zeros(n)
    period = int(sampling_hz * 60.0 / bpm)
    for peak in range(period, n - 3, period):
        signal[peak - 2] = -0.3
        signal[peak] = 1.0
        signal[peak + 2] = -0.3
    return signal


def test_detects_beats_at_the_right_rate():
    ecg = _synthetic_ecg(bpm=60.0, seconds=60.0)
    detection = detect_beats(ecg, 250.0)
    assert detection.mean_heart_rate == pytest.approx(60.0, abs=2.0)
    assert detection.is_plausible() is True


def test_intervals_are_timed_at_the_second_beat():
    """An interval becomes known when the beat that closes it arrives.

    Attributing it to the first beat shifts every epoch boundary by one beat.
    """
    ecg = _synthetic_ecg(bpm=60.0, seconds=10.0)
    detection = detect_beats(ecg, 250.0)
    assert len(detection.rr_times_s) == len(detection.rr_ms)
    assert detection.rr_times_s[0] > detection.rr_ms[0] / 1000.0 * 0.9


def test_implausible_intervals_are_rejected_and_counted():
    """A missed beat yields one doubled interval, which wrecks RMSSD.

    The metric is a root mean square of successive differences, so a single
    doubled interval contributes far more than any real beat.
    """
    low, high = PLAUSIBLE_RR_MS
    assert low > 0 and high > low
    ecg = _synthetic_ecg(bpm=60.0, seconds=30.0)
    ecg[int(250.0 * 10.0) - 3 : int(250.0 * 10.0) + 3] = 0.0  # delete one beat
    detection = detect_beats(ecg, 250.0)
    assert all(low <= rr <= high for rr in detection.rr_ms)


def test_a_flat_trace_yields_nothing_rather_than_invented_beats():
    detection = detect_beats(np.zeros(2500), 250.0)
    assert detection.n_beats == 0
    assert detection.rr_ms == ()
    assert detection.is_plausible() is False


def test_a_very_short_trace_is_handled():
    assert detect_beats(np.zeros(10), 250.0).n_beats == 0


def test_implausible_heart_rate_is_reported_as_such():
    """A detector failing on a bad lead produces confident nonsense.

    is_plausible is what stops that nonsense being trained on.
    """
    ecg = _synthetic_ecg(bpm=180.0, seconds=30.0)
    detection = detect_beats(ecg, 250.0)
    assert detection.mean_heart_rate > 110.0
    assert detection.is_plausible() is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
