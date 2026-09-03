"""Store behaviour, especially idempotency on replayed BLE buffers."""

import pytest

from vitalgraph.biometrics.schema import InvalidSample, Sample, SignalType, utc
from vitalgraph.biometrics.store import BiometricStore


@pytest.fixture()
def store():
    with BiometricStore(":memory:") as s:
        yield s


def _rr(ts, value):
    return Sample(ts=utc(ts), signal=SignalType.RR_INTERVAL, value=value, source="test")


def test_add_and_read_back(store):
    store.add([_rr(0, 800.0), _rr(1, 810.0)])
    assert store.rr_series(utc(0), utc(10)) == [800.0, 810.0]


def test_reinserting_the_same_beats_does_not_double_count(store):
    """A BLE reconnect commonly replays an overlapping buffer."""
    batch = [_rr(0, 800.0), _rr(1, 810.0)]
    store.add(batch)
    store.add(batch)
    assert store.count(SignalType.RR_INTERVAL) == 2


def test_window_is_half_open(store):
    store.add([_rr(0, 800.0), _rr(10, 810.0)])
    assert store.rr_series(utc(0), utc(10)) == [800.0]


def test_results_are_time_ordered_regardless_of_insert_order(store):
    store.add([_rr(5, 850.0), _rr(1, 800.0), _rr(3, 820.0)])
    assert store.rr_series(utc(0), utc(10)) == [800.0, 820.0, 850.0]


def test_span_and_signals(store):
    store.add([_rr(0, 800.0), _rr(60, 810.0),
               Sample(ts=utc(30), signal=SignalType.SPO2, value=97.0)])
    lo, hi = store.span()
    assert lo == utc(0) and hi == utc(60)
    assert set(store.signals()) == {"rr_interval", "spo2"}


def test_empty_store_reports_no_span(store):
    assert store.span() is None
    assert store.count() == 0


def test_implausible_values_are_rejected_at_the_door():
    with pytest.raises(InvalidSample):
        Sample(ts=utc(0), signal=SignalType.SPO2, value=140.0)


def test_naive_timestamps_are_rejected():
    from datetime import datetime

    with pytest.raises(InvalidSample):
        Sample(ts=datetime(2026, 1, 1), signal=SignalType.HEART_RATE, value=60.0)


def test_store_is_usable_from_multiple_threads(store):
    """Regression: FastAPI runs sync endpoints on a threadpool, so a store
    confined to its creating thread raises ProgrammingError under load."""
    import threading

    errors = []

    def writer(offset):
        try:
            store.add([_rr(offset * 100 + i, 800.0 + i) for i in range(50)])
            store.rr_series(utc(0), utc(100000))
            store.count()
        except Exception as exc:  # pragma: no cover - only on regression
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert store.count(SignalType.RR_INTERVAL) == 8 * 50
