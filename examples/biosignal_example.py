"""End-to-end walkthrough of the BLE biosignal subsystem.

Runs with no hardware and no API keys: it decodes hand-assembled GATT packets,
merges in a vendor cloud payload and a phone health-store export, runs the
analysis chain, and prints the content list that would be indexed into
RAG-Anything.

    python examples/biosignal_example.py

The final section shows the live-capture and real-indexing paths, which need
``bleak`` and a configured ``RAGAnything`` respectively.
"""

import asyncio
import json
import math
import struct
import time

from raganything.biosignal import narrative, sources
from raganything.biosignal.query import BiosignalQueryEngine
from raganything.biosignal.store import ReportStore
from raganything.biosignal.ble import codecs, uuids
from raganything.biosignal.ble.client import StreamRecorder
from raganything.biosignal.schema import Session


def synth_heart_rate_packet(bpm: int, contact: bool = True) -> bytes:
    """Build a Heart Rate Measurement packet with RR intervals, as a strap would."""
    flags = 0x10  # RR intervals present
    flags |= 0x04  # sensor contact supported
    if contact:
        flags |= 0x02  # ...and detected

    rr_ms = 60000.0 / bpm
    intervals = []
    for i in range(3):
        # A little respiratory sinus arrhythmia so HRV is not identically zero.
        jitter = 18.0 * math.sin(i * 1.1 + bpm)
        intervals.append(struct.pack("<H", int((rr_ms + jitter) * 1024 / 1000)))
    return bytes([flags, bpm]) + b"".join(intervals)


def synth_indoor_bike_packet(speed_kph: float, cadence: float, watts: int) -> bytes:
    """Build an FTMS Indoor Bike Data packet, as a smart trainer would."""
    flags = 0x0044  # bit0 clear: speed present; bit2 cadence; bit6 power
    return (
        struct.pack("<H", flags)
        + struct.pack("<H", int(speed_kph * 100))
        + struct.pack("<H", int(cadence * 2))
        + struct.pack("<h", watts)
    )


def build_session_from_packets() -> Session:
    """Replay twenty minutes of notifications through the decoders."""
    start = time.time() - 1200
    strap = StreamRecorder(source_id="chest-strap", device="Polar H10")
    trainer = StreamRecorder(source_id="trainer", device="Wahoo KICKR")

    for second in range(1200):
        t = start + second
        # Warm up, work, then fade -- so decoupling has something to find.
        if second < 300:
            bpm, watts = 120 + second // 30, 160
        elif second < 900:
            bpm, watts = 152 + (second % 7), 240
        else:
            bpm, watts = 158 + (second % 5), 225

        # A dropout: the strap loses skin contact for ninety seconds.
        contact = not (600 <= second < 690)

        if second % 1 == 0:
            strap.ingest(
                codecs.decode(
                    uuids.CHR_HEART_RATE_MEASUREMENT,
                    synth_heart_rate_packet(bpm, contact=contact),
                    t,
                )
            )
        if second % 2 == 0:
            trainer.ingest(
                codecs.decode(
                    uuids.CHR_INDOOR_BIKE_DATA,
                    synth_indoor_bike_packet(32.0, 88.0, watts),
                    t,
                )
            )

    session = strap.to_session("ride-2026-09-02", start=start, end=start + 1200)
    for stream in trainer.to_session("trainer", start=start, end=start + 1200).streams:
        session.add(stream)
    session.labels.update({"sport": "cycling", "context": "indoor threshold session"})
    return session


def add_other_sources(session: Session) -> None:
    """Fold in a vendor cloud payload and a phone health-store export.

    The watch's optical heart rate is deliberately biased and lagged relative to
    the strap, which is what a wrist sensor actually does under load. Fusion
    reports that disagreement instead of silently preferring one series.
    """
    from datetime import datetime, timezone

    def iso(offset_s: float) -> str:
        return datetime.fromtimestamp(
            session.start + offset_s, tz=timezone.utc
        ).isoformat()

    health_records = [
        {
            "type": "HeartRateRecord",
            "startTime": iso(second),
            # Optical wrist HR reads high and late during hard efforts.
            "value": 120 + second // 30 + 9 if second < 300 else 161 + (second % 4),
            "dataOrigin": "com.watch.health",
        }
        for second in range(0, 1200, 5)
    ]
    for stream in sources.normalize_health_records(health_records):
        session.add(stream)

    # A morning readiness score from a ring: one value, six hours stale by the
    # time it is available, and not recomputable from anything the API returns.
    oura_payload = {"data": [{"timestamp": iso(-18000), "score": 71}]}
    for stream in sources.normalize_vendor_payload("oura", oura_payload):
        session.add(stream)


def main() -> None:
    session = build_session_from_packets()
    add_other_sources(session)

    print("=" * 78)
    print("SESSION")
    print("=" * 78)
    print(json.dumps(session.to_dict(), indent=2, default=str)[:1600])

    # Athlete constants are passed explicitly. Omit them and the dependent
    # metrics come back as withheld rather than as a guess about the athlete.
    report = narrative.analyze_session(
        session, rest_hr=48, max_hr=186, threshold_power=250.0, sex="unspecified"
    )

    print()
    print("=" * 78)
    print("REPORTED METRICS")
    print("=" * 78)
    for name, value in sorted(report.metrics.items()):
        print(f"  {name:<26} {value:>10.2f}")

    print()
    print("=" * 78)
    print("WITHHELD -- and why")
    print("=" * 78)
    for name, reason in sorted(report.withheld.items()):
        print(f"  {name}\n      {reason}")

    if report.hrv is not None:
        print()
        print("=" * 78)
        print("HRV PROVENANCE")
        print("=" * 78)
        print(json.dumps(report.hrv.to_dict(), indent=2))

    print()
    print("=" * 78)
    print("SIGNAL QUALITY")
    print("=" * 78)
    for source_id, qr in report.quality.items():
        print(f"  {source_id:<34} score={qr.score:.2f} coverage={qr.coverage:.1%}")
        for reason in qr.reasons:
            print(f"      - {reason}")

    print()
    print("=" * 78)
    print("CONTENT LIST FOR THE KNOWLEDGE GRAPH")
    print("=" * 78)
    for item in narrative.to_content_list(report):
        print(f"\n--- page {item['page_idx']} ({item['type']}) ---")
        print(item.get("text") or item.get("table_body"))


def query_layer_example() -> None:
    """Answer questions over a history of sessions, with no LLM involved.

    Everything below is arithmetic over the stored reports. The withheld
    metrics and excluded sessions are printed alongside the answers, because
    the denominator is part of the answer.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        store = ReportStore(tmp)
        base = time.time() - 45 * 86400

        for i in range(14):
            start = base + i * 3 * 86400
            # A gently declining HRV, and one session where the strap fails.
            amplitude = 30.0 - i * 1.1
            broken = i == 9
            rr = []
            for k in range(900):
                t = start + k * 0.6
                rr.append((t, 1000.0 + (amplitude if k % 2 else -amplitude)))
            streams = []
            if not broken:
                streams.append(_rr_stream(rr))
            streams.append(_hr_stream(start, 138.0 + i * 0.4))
            session = Session(
                f"ride-{i:02d}",
                start,
                start + 540,
                streams=streams,
                labels={"sport": "cycling"},
            )
            store.put(narrative.analyze_session(session))

        engine = BiosignalQueryEngine(store=store)
        questions = [
            "is my HRV trending down over the last six weeks?",
            "what was my average heart rate over the last 6 weeks?",
            "what was my highest heart rate last month?",
        ]
        for question in questions:
            answer = engine.compute(question)
            print()
            print("=" * 78)
            print("Q:", question)
            print("=" * 78)
            print(answer.answer)
            if answer.withheld:
                print("\n  withheld:")
                for key, reason in list(answer.withheld.items())[:3]:
                    print(f"    {key}: {reason[:88]}")


def _rr_stream(pairs):
    from raganything.biosignal.schema import (
        Modality,
        Provenance,
        SourceKind,
        make_stream,
    )

    return make_stream(
        Modality.RR_INTERVAL,
        pairs,
        Provenance(
            source_id="strap-rr",
            kind=SourceKind.BLE,
            device="Polar H10",
            latency_s=0.05,
            nominal_hz=2.0,
        ),
    )


def _hr_stream(start, bpm):
    from raganything.biosignal.schema import (
        Modality,
        Provenance,
        SourceKind,
        make_stream,
    )

    return make_stream(
        Modality.HEART_RATE,
        [(start + i, bpm) for i in range(540)],
        Provenance(
            source_id="strap-hr",
            kind=SourceKind.BLE,
            device="Polar H10",
            latency_s=0.05,
            nominal_hz=1.0,
        ),
    )


async def live_capture_example() -> None:  # pragma: no cover - needs hardware
    """Capture from a real device. Requires ``pip install raganything[biosignal]``."""
    from raganything.biosignal.ble.client import BLECollector

    for device in await BLECollector.scan(timeout=8.0):
        print(device["name"], device["address"], device["supported_services"])

    collector = BLECollector(address="XX:XX:XX:XX:XX:XX", device_name="Polar H10")
    session = await collector.collect(duration_s=300)
    report = narrative.analyze_session(session)
    print(report.to_dict())


async def indexing_example() -> None:  # pragma: no cover - needs an LLM backend
    """Insert a session into a configured RAG-Anything knowledge graph."""
    from raganything import RAGAnything, RAGAnythingConfig
    from raganything.biosignal.index import index_session

    rag = RAGAnything(config=RAGAnythingConfig(working_dir="./rag_storage"))
    session = build_session_from_packets()
    report = await index_session(rag, session, rest_hr=48, max_hr=186)
    print(f"indexed {session.session_id}: {len(report.metrics)} metrics")

    answer = await rag.aquery(
        "During my last ride, was the heart rate data reliable enough to trust "
        "the HRV figure? What went wrong, if anything?"
    )
    print(answer)


if __name__ == "__main__":
    main()
    print("\n\n")
    print("#" * 78)
    print("# QUERY LAYER -- deterministic answers, no LLM and no network")
    print("#" * 78)
    query_layer_example()
    if False:  # flip to True with hardware / a configured backend
        asyncio.run(live_capture_example())
        asyncio.run(indexing_example())
