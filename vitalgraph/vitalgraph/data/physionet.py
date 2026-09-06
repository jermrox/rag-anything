"""Reading real polysomnography from PhysioNet, so the stager can be scored
against something other than a simulator.

The stager reports itself as leaking -- 100% accuracy on every leave-one-night-
out fold -- and no amount of further metric work fixes that, because the cause
is the training data. The simulator's stage track is a deterministic function
of elapsed time, so a model given time-correlated features recovers it exactly.
The only repair is real nights from real people.

**Which dataset, and why not the obvious one.** Sleep-EDF is the best-known
open sleep corpus and is the wrong choice here: it is EEG-based and carries no
ECG, so it yields no RR intervals, which is the entire input to our stager.
The MIT-BIH Polysomnographic Database (``slpdb``) does carry ECG at 250 Hz
alongside expert sleep-stage annotations, which is exactly the pairing needed
to score a wrist-style stager honestly.

**Why a reader rather than a dependency.** ``wfdb`` would read this, and it is
in the harvested corpus under MIT, so its formats were free to learn from. But
it pulls pandas, matplotlib and soundfile for what is, here, two well-specified
binary layouts totalling about a hundred lines. Format 212 packs two 12-bit
samples into three bytes; the annotation format is a 16-bit word of type code
and sample interval. Both are documented, both are stable, and implementing
them keeps the dependency surface honest.

Nothing here is committed to the repository. Records are cached under
``~/.vitalgraph/physionet`` and downloaded on demand, so the corpus stays out
of version control and the licence stays with PhysioNet.
"""

from __future__ import annotations

import struct
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from ..biometrics.schema import SleepStage

#: Base URL for the MIT-BIH Polysomnographic Database.
SLPDB_BASE = "https://physionet.org/files/slpdb/1.0.0"

#: Where downloaded records are cached. Outside the repository on purpose.
DEFAULT_CACHE = Path.home() / ".vitalgraph" / "physionet"

#: Records carrying both ECG and sleep-stage annotations. Not the whole
#: database: several records are ECG-only or annotation-only, and a record
#: missing either is useless for this purpose.
SLPDB_RECORDS: Tuple[str, ...] = (
    "slp01a",
    "slp01b",
    "slp02a",
    "slp02b",
    "slp03",
    "slp04",
    "slp14",
    "slp16",
    "slp32",
    "slp37",
    "slp41",
    "slp45",
    "slp48",
    "slp59",
    "slp60",
    "slp61",
    "slp66",
    "slp67x",
)

#: Network retry schedule, matching the harvester's.
RETRY_BACKOFF_S = (2.0, 4.0, 8.0, 16.0)

#: An epoch is 30 seconds, the standard scoring window and the same one the
#: simulator and the epoch builder use.
EPOCH_SECONDS = 30.0

#: slpdb stage codes mapped onto our four-class model.
#:
#: Stages 1 and 2 collapse to LIGHT and 3 and 4 to DEEP, which is the
#: conventional reduction and also the only honest one for a wrist device:
#: separating N1 from N2 needs EEG, and claiming to do it from RR intervals
#: would be exactly the overclaim the signal model exists to prevent.
STAGE_CODES: Dict[str, SleepStage] = {
    "W": SleepStage.AWAKE,
    "1": SleepStage.LIGHT,
    "2": SleepStage.LIGHT,
    "3": SleepStage.DEEP,
    "4": SleepStage.DEEP,
    "R": SleepStage.REM,
}


class DownloadError(RuntimeError):
    """A record could not be fetched."""


class FormatError(ValueError):
    """A file did not match the WFDB layout it claims."""


@dataclass(frozen=True, slots=True)
class SignalSpec:
    """One channel from a WFDB header."""

    filename: str
    fmt: str
    gain: float
    """ADC units per physical unit. Zero in a header means 200 by convention,
    and dividing by a literal zero here would be a crash on a valid file."""

    baseline: int
    name: str


@dataclass(frozen=True, slots=True)
class RecordHeader:
    """A parsed WFDB ``.hea``."""

    name: str
    n_signals: int
    sampling_hz: float
    n_samples: int
    signals: Tuple[SignalSpec, ...]

    def channel_index(self, name: str) -> int | None:
        """Index of a channel by name, matched loosely.

        Channel names vary across records -- "ECG", "ECG1", "ECG (V5)" -- so an
        exact match would silently find nothing on half the database and return
        an empty result rather than an error.
        """
        wanted = name.strip().lower()
        for index, signal in enumerate(self.signals):
            if signal.name.strip().lower().startswith(wanted):
                return index
        return None


def _fetch(url: str, dest: Path, sleep=None) -> Path:
    """Download ``url`` to ``dest``, retrying on transient failure."""
    import time

    sleep = sleep or time.sleep
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)

    last = ""
    for attempt in range(len(RETRY_BACKOFF_S) + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                payload = response.read()
            if not payload:
                raise DownloadError("empty response")
            dest.write_bytes(payload)
            return dest
        except (urllib.error.URLError, OSError, DownloadError) as exc:
            last = str(exc)
            if attempt < len(RETRY_BACKOFF_S):
                sleep(RETRY_BACKOFF_S[attempt])
    raise DownloadError(f"failed to download {url}: {last}")


def parse_header(text: str) -> RecordHeader:
    """Parse a WFDB ``.hea`` file.

    Only the fields this module uses are read. Anything unrecognised is
    ignored rather than guessed at, since a misread gain silently rescales
    every sample.
    """
    lines = [
        line
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not lines:
        raise FormatError("empty header")

    fields = lines[0].split()
    if len(fields) < 3:
        raise FormatError(f"malformed header line: {lines[0]!r}")
    name = fields[0]
    n_signals = int(fields[1])
    # The frequency field may carry a counter spec, e.g. "250/0.033(94)".
    sampling_hz = float(fields[2].split("/")[0])
    n_samples = int(fields[3]) if len(fields) > 3 else 0

    signals: List[SignalSpec] = []
    for line in lines[1 : 1 + n_signals]:
        parts = line.split()
        if len(parts) < 2:
            raise FormatError(f"malformed signal line: {line!r}")
        gain_field = parts[2] if len(parts) > 2 else "200"
        gain_text = gain_field.split("/")[0].split("(")[0]
        try:
            gain = float(gain_text)
        except ValueError:
            gain = 200.0
        # A gain of zero means "unspecified", which the specification defines
        # as 200 ADC units per mV. Taking it literally divides by zero.
        if gain == 0.0:
            gain = 200.0
        baseline = 0
        if len(parts) > 4:
            try:
                baseline = int(parts[4])
            except ValueError:
                baseline = 0
        signal_name = " ".join(parts[8:]) if len(parts) > 8 else f"sig{len(signals)}"
        signals.append(
            SignalSpec(
                filename=parts[0],
                fmt=parts[1],
                gain=gain,
                baseline=baseline,
                name=signal_name,
            )
        )

    return RecordHeader(
        name=name,
        n_signals=n_signals,
        sampling_hz=sampling_hz,
        n_samples=n_samples,
        signals=tuple(signals),
    )


def decode_format_212(data: bytes, n_signals: int, n_samples: int):
    """Decode WFDB format 212: two 12-bit samples packed into three bytes.

    Byte layout, little-endian nibbles::

        byte 0        low 8 bits of sample A
        byte 1        low nibble  = high 4 bits of sample A
                      high nibble = high 4 bits of sample B
        byte 2        low 8 bits of sample B

    Both samples are 12-bit two's complement, so values above 2047 are
    negative. Skipping that conversion leaves every negative deflection of the
    ECG reading as a large positive number, which inverts the QRS complex and
    quietly breaks peak detection rather than raising anything.
    """
    import numpy as np

    total = n_signals * n_samples
    triples = (total + 1) // 2
    needed = triples * 3
    if len(data) < needed:
        triples = len(data) // 3
        total = triples * 2

    raw = np.frombuffer(data[: triples * 3], dtype=np.uint8).reshape(-1, 3)
    b0 = raw[:, 0].astype(np.int32)
    b1 = raw[:, 1].astype(np.int32)
    b2 = raw[:, 2].astype(np.int32)

    first = ((b1 & 0x0F) << 8) | b0
    second = ((b1 & 0xF0) << 4) | b2

    samples = np.empty(triples * 2, dtype=np.int32)
    samples[0::2] = first
    samples[1::2] = second
    samples = samples[:total]

    # 12-bit two's complement.
    samples = np.where(samples > 2047, samples - 4096, samples)

    usable = (len(samples) // n_signals) * n_signals
    return samples[:usable].reshape(-1, n_signals)


def parse_annotations(data: bytes) -> List[Tuple[int, str]]:
    """Decode a WFDB annotation file into ``(sample, aux_string)`` pairs.

    Each annotation is a 16-bit word: the top six bits are a type code and the
    low ten bits the sample interval since the previous annotation. Codes 59
    to 63 are escapes carrying extra data; only SKIP (59) and AUX (63) affect
    the result here, and the rest are consumed so their payloads are not
    mistaken for annotations.
    """
    out: List[Tuple[int, str]] = []
    i = 0
    t = 0
    pending_aux: str | None = None

    while i + 1 < len(data):
        word = struct.unpack_from("<H", data, i)[0]
        i += 2
        code = word >> 10
        interval = word & 0x03FF

        if code == 0 and interval == 0:
            break

        if code == 63:  # AUX: interval is a byte count, padded to even.
            length = interval
            pending_aux = data[i : i + length].decode("ascii", "replace")
            i += length + (length % 2)
            if out:
                sample, _ = out[-1]
                out[-1] = (sample, pending_aux)
            continue

        if code == 59:  # SKIP: a 32-bit interval follows, high word first.
            if i + 6 > len(data):
                break
            high, low = struct.unpack_from("<HH", data, i)
            i += 4
            t += (high << 16) | low
            i += 2  # the annotation word that follows the escape
            out.append((t, ""))
            continue

        if code in (60, 61, 62):  # NUM, SUB, CHN carry no sample interval.
            continue

        t += interval
        out.append((t, ""))

    return out


def stages_from_annotations(
    annotations: Sequence[Tuple[int, str]], sampling_hz: float
) -> List[Tuple[int, SleepStage]]:
    """Turn annotations into ``(epoch_index, stage)`` pairs.

    The aux string is the stage code followed by any events scored in that
    epoch -- ``"2 OA"`` is stage 2 with an obstructive apnoea. Only the first
    token is the stage; the events are a different question and are dropped
    here rather than silently folded into the label.

    Epochs whose code is not a sleep stage (movement time, or an unscored
    epoch) are omitted, so an unscorable epoch never becomes a wrong label.
    """
    per_epoch = int(EPOCH_SECONDS * sampling_hz)
    if per_epoch <= 0:
        raise FormatError(f"nonsensical sampling rate {sampling_hz}")

    staged: List[Tuple[int, SleepStage]] = []
    for sample, aux in annotations:
        if not aux:
            continue
        code = aux.split()[0] if aux.split() else ""
        stage = STAGE_CODES.get(code)
        if stage is None:
            continue
        staged.append((sample // per_epoch, stage))
    return staged


@dataclass(frozen=True, slots=True)
class PolysomnographyRecord:
    """One real night: an ECG trace and the stages an expert scored."""

    name: str
    sampling_hz: float
    ecg: object
    """1-D numpy array of ECG in millivolts."""

    stages: Tuple[Tuple[int, SleepStage], ...]
    """``(epoch_index, stage)``, 30-second epochs."""

    @property
    def duration_hours(self) -> float:
        return len(self.ecg) / self.sampling_hz / 3600.0

    @property
    def n_staged_epochs(self) -> int:
        return len(self.stages)

    def summary(self) -> Dict[str, object]:
        counts: Dict[str, int] = {}
        for _, stage in self.stages:
            counts[stage.name.lower()] = counts.get(stage.name.lower(), 0) + 1
        return {
            "record": self.name,
            "hours": round(self.duration_hours, 2),
            "sampling_hz": self.sampling_hz,
            "staged_epochs": self.n_staged_epochs,
            "stage_counts": counts,
        }


def load_record(
    record: str,
    cache_dir: str | Path = DEFAULT_CACHE,
    base_url: str = SLPDB_BASE,
) -> PolysomnographyRecord:
    """Download if needed, then read one record's ECG and staging."""
    cache = Path(cache_dir)
    header = parse_header(
        _fetch(f"{base_url}/{record}.hea", cache / f"{record}.hea").read_text(
            encoding="utf-8", errors="replace"
        )
    )

    ecg_index = header.channel_index("ecg")
    if ecg_index is None:
        raise FormatError(
            f"{record} has no ECG channel; channels are "
            f"{[s.name for s in header.signals]}"
        )
    spec = header.signals[ecg_index]
    if spec.fmt != "212":
        raise FormatError(f"{record} uses format {spec.fmt}; only 212 is supported")

    raw = _fetch(f"{base_url}/{record}.dat", cache / f"{record}.dat").read_bytes()
    matrix = decode_format_212(raw, header.n_signals, header.n_samples)
    ecg = (matrix[:, ecg_index] - spec.baseline) / spec.gain

    annotation_bytes = _fetch(
        f"{base_url}/{record}.st", cache / f"{record}.st"
    ).read_bytes()
    stages = stages_from_annotations(
        parse_annotations(annotation_bytes), header.sampling_hz
    )

    return PolysomnographyRecord(
        name=record,
        sampling_hz=header.sampling_hz,
        ecg=ecg,
        stages=tuple(stages),
    )
