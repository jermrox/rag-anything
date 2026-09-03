"""Time-series persistence for biometric samples.

SQLite via the standard library. It is dependency-free, transactional, and
handles the M1 volumes (~10^5-10^7 samples) comfortably with the right index.
The read path is deliberately narrow -- ``samples_in`` and ``rr_series`` -- so
swapping in DuckDB/Parquet later (M4, for multi-year columnar aggregation)
touches only this file.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence

from .schema import Sample, SignalType, utc

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    ts_ms   INTEGER NOT NULL,
    signal  TEXT    NOT NULL,
    value   REAL    NOT NULL,
    source  TEXT    NOT NULL DEFAULT 'unknown',
    PRIMARY KEY (ts_ms, signal, source)
) WITHOUT ROWID;

-- Every read is "one signal over a time window", so lead with signal.
CREATE INDEX IF NOT EXISTS idx_samples_signal_ts ON samples (signal, ts_ms);
"""


class BiometricStore:
    """Append-oriented store for canonical :class:`Sample` records."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)

        # A sqlite3 connection is bound to its creating thread by default, but
        # FastAPI dispatches synchronous endpoints onto a threadpool -- so a
        # single shared connection raises ProgrammingError under normal server
        # load. Opening with check_same_thread=False and serialising every
        # access through one lock keeps the connection usable from any worker
        # thread. A single user's biometric stream is nowhere near lock-bound,
        # and this keeps ``:memory:`` working (per-thread connections would
        # each get their own empty database).
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "BiometricStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def add(self, samples: Iterable[Sample]) -> int:
        """Insert samples idempotently.

        Uses INSERT OR IGNORE so replaying an overlapping BLE buffer -- which
        happens routinely on reconnect -- cannot double-count beats.
        """
        rows = [(s.epoch_ms, s.signal.value, s.value, s.source) for s in samples]
        if not rows:
            return 0
        with self._lock:
            with closing(self._conn.cursor()) as cur:
                cur.executemany(
                    "INSERT OR IGNORE INTO samples (ts_ms, signal, value, source)"
                    " VALUES (?, ?, ?, ?)",
                    rows,
                )
                inserted = cur.rowcount
            self._conn.commit()
        return max(inserted, 0)

    def samples_in(
        self, signal: SignalType, start: datetime, end: datetime
    ) -> List[Sample]:
        """All samples of ``signal`` in the half-open interval [start, end)."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT ts_ms, value, source FROM samples"
                " WHERE signal = ? AND ts_ms >= ? AND ts_ms < ? ORDER BY ts_ms",
                (signal.value, int(start.timestamp() * 1000), int(end.timestamp() * 1000)),
            )
            rows = cur.fetchall()
        return [
            Sample(ts=utc(ts_ms / 1000.0), signal=signal, value=value, source=source)
            for ts_ms, value, source in rows
        ]

    def rr_series(self, start: datetime, end: datetime) -> List[float]:
        """RR intervals (ms) in a window -- the input to every HRV metric."""
        with self._lock:
            cur = self._conn.execute(
                "SELECT value FROM samples WHERE signal = ? AND ts_ms >= ? AND ts_ms < ?"
                " ORDER BY ts_ms",
                (
                    SignalType.RR_INTERVAL.value,
                    int(start.timestamp() * 1000),
                    int(end.timestamp() * 1000),
                ),
            )
            return [row[0] for row in cur.fetchall()]

    def span(self) -> tuple[datetime, datetime] | None:
        """Earliest and latest sample timestamps, or None when empty."""
        with self._lock:
            row = self._conn.execute(
                "SELECT MIN(ts_ms), MAX(ts_ms) FROM samples"
            ).fetchone()
        if not row or row[0] is None:
            return None
        return utc(row[0] / 1000.0), utc(row[1] / 1000.0)

    def count(self, signal: SignalType | None = None) -> int:
        with self._lock:
            if signal is None:
                row = self._conn.execute("SELECT COUNT(*) FROM samples").fetchone()
            else:
                row = self._conn.execute(
                    "SELECT COUNT(*) FROM samples WHERE signal = ?", (signal.value,)
                ).fetchone()
        return int(row[0])

    def signals(self) -> Sequence[str]:
        with self._lock:
            return [
                r[0] for r in self._conn.execute("SELECT DISTINCT signal FROM samples")
            ]
