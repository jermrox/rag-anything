"""FastAPI backend for VitalGraph.

Serves the single-file web UI, accepts decoded BLE samples pushed from the
browser's Web Bluetooth session, exposes computed metrics, and proxies RAG
queries. RAG endpoints degrade to a clear 503 when ``raganything`` or an LLM
key is absent, so the analytics half of the product stays usable on its own.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from ..ble import gatt
from ..ble.simulator import simulate_period
from ..biometrics import hrv
from ..biometrics.schema import InvalidSample, Sample, SignalType, utc
from ..biometrics.store import BiometricStore
from ..bridge import summarizer as S
from ..config import VitalGraphConfig
from ..rag import RAGAnythingUnavailable, VitalGraphRAG

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

config = VitalGraphConfig()
Path(config.data_dir).mkdir(parents=True, exist_ok=True)
store = BiometricStore(config.db_path)

app = FastAPI(
    title="VitalGraph",
    description="Health analytics and BLE knowledge RAG",
    version="0.1.0",
)

_rag: Optional[VitalGraphRAG] = None


def get_rag() -> VitalGraphRAG:
    global _rag
    if _rag is None:
        try:
            _rag = VitalGraphRAG.from_env(config.rag_working_dir)
        except RAGAnythingUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return _rag


# --- Request models --------------------------------------------------------


class SamplePayload(BaseModel):
    ts_ms: int = Field(..., description="Epoch milliseconds, UTC")
    signal: str
    value: float
    source: str = "ble"


class StreamPayload(BaseModel):
    """Decoded samples pushed from a Web Bluetooth session."""

    samples: List[SamplePayload]


class GattPayload(BaseModel):
    """A raw 0x2A37 notification, hex-encoded, decoded server-side."""

    hex: str
    ts_ms: Optional[int] = None
    source: str = "ble:0x2A37"


class QueryPayload(BaseModel):
    question: str
    mode: str = "mix"


class SeedPayload(BaseModel):
    nights: int = 7
    recovery: Optional[List[float]] = None
    seed: int = 11


# --- Routes ----------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> Dict[str, Any]:
    span = store.span()
    return {
        "status": "ok",
        "samples": store.count(),
        "signals": sorted(store.signals()),
        "span": [span[0].isoformat(), span[1].isoformat()] if span else None,
        "rag_available": _rag is not None,
    }


@app.post("/api/ingest/stream")
def ingest_stream(payload: StreamPayload) -> Dict[str, int]:
    """Ingest decoded samples (the Web Bluetooth path)."""
    samples: List[Sample] = []
    for item in payload.samples:
        try:
            samples.append(
                Sample(
                    ts=utc(item.ts_ms / 1000.0),
                    signal=SignalType(item.signal),
                    value=item.value,
                    source=item.source,
                )
            )
        except (InvalidSample, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"{item.signal}: {exc}") from exc
    return {"inserted": store.add(samples), "received": len(samples)}


@app.post("/api/ingest/gatt")
def ingest_gatt(payload: GattPayload) -> Dict[str, Any]:
    """Ingest a raw GATT Heart Rate Measurement notification.

    Accepting the raw payload -- not just a decoded number -- means the server
    keeps the authoritative decoder, so every client benefits from fixes to it.
    """
    try:
        raw = bytes.fromhex(payload.hex.replace(" ", ""))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid hex: {exc}") from exc
    try:
        measurement = gatt.decode_heart_rate_measurement(raw)
    except gatt.DecodeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ts_ms = payload.ts_ms or int(datetime.now(timezone.utc).timestamp() * 1000)
    base = utc(ts_ms / 1000.0)
    samples = [
        Sample(ts=base, signal=SignalType.HEART_RATE,
               value=float(measurement.heart_rate_bpm), source=payload.source)
    ]
    # RR intervals are stamped backwards from the notification time, since they
    # describe beats that already happened.
    offset = 0.0
    for rr in reversed(measurement.rr_intervals_ms):
        offset += rr
        samples.append(
            Sample(ts=base - timedelta(milliseconds=offset),
                   signal=SignalType.RR_INTERVAL, value=rr, source=payload.source)
        )
    return {
        "inserted": store.add(samples),
        "heart_rate": measurement.heart_rate_bpm,
        "rr_count": len(measurement.rr_intervals_ms),
    }


@app.get("/api/metrics")
def metrics(nights: int = 7, hour: int = 0) -> Dict[str, Any]:
    """Nightly summaries with a rolling personal baseline."""
    span = store.span()
    if span is None:
        return {"periods": [], "message": "No data yet. Seed the demo or connect a device."}

    first = span[0].replace(hour=hour, minute=0, second=0, microsecond=0)
    available = int((span[1] - first).total_seconds() // 86400) + 1
    windows = S.nightly_windows(first, min(nights, available), hour=hour)

    baseline: List[float] = []
    out = []
    for pid, start, end in windows:
        summ = S.summarize_period(
            store, start, end, pid, user=config.user,
            rmssd_baseline=baseline[-config.baseline_nights:],
        )
        # A window without enough beats has no computable HRV. Emitting it
        # would put a 0 ms point on the chart, and "no data" rendered as zero
        # reads as "no variability" -- a materially wrong clinical impression.
        # Partial windows are common at the edges: a night in progress, or a
        # single reconnect notification landing in a fresh day.
        if summ.metrics.n_beats < hrv.MIN_BEATS:
            continue
        baseline.append(summ.metrics.rmssd_ms)
        out.append(
            {
                "period_id": summ.period_id,
                "date": summ.start.strftime("%Y-%m-%d"),
                "citation": summ.citation_ref,
                "verdict": summ.verdict,
                "metrics": summ.metrics.as_dict(),
                "stage_minutes": summ.stage_minutes,
                "sleep_minutes": summ.sleep_minutes,
                "mean_spo2": summ.mean_spo2,
                "mean_skin_temp": summ.mean_skin_temp,
                "rmssd_baseline": summ.rmssd_baseline,
                "rmssd_z": summ.rmssd_z,
                "narrative": S.render_narrative(summ),
            }
        )
    return {"periods": out}


@app.get("/api/protocol/derivable")
def protocol_derivable(uuids: str) -> Dict[str, Any]:
    """What health signals a device's characteristics make computable.

    ``uuids`` is a comma-separated list, e.g. ``0x2A37,0x2A19``.
    """
    requested = [u.strip() for u in uuids.split(",") if u.strip()]
    derivable = gatt.derivable_signals(requested)
    return {
        "requested": requested,
        "derivable": derivable,
        "unrecognised": [
            u for u in requested if u.upper().replace("0X", "0x") not in derivable
        ],
    }


@app.post("/api/demo/seed")
def demo_seed(payload: SeedPayload) -> Dict[str, Any]:
    """Populate the store with simulated nights.

    Defaults to a scripted narrative -- good nights followed by a sharp
    decline -- so the RAG has something worth being asked about.
    """
    nights = max(1, min(payload.nights, 60))
    recovery = payload.recovery or (
        [1.0, 0.95, 0.9, 1.0, 0.95, 0.9, 1.0, 0.95, 0.25, 0.2][:nights]
        if nights <= 10
        else [1.0] * (nights - 2) + [0.25, 0.2]
    )
    if len(recovery) != nights:
        recovery = (recovery + [1.0] * nights)[:nights]

    start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ) - timedelta(days=nights)
    inserted = store.add(
        simulate_period(start, nights=nights, recovery_by_night=recovery, seed=payload.seed)
    )
    return {"inserted": inserted, "nights": nights, "recovery": recovery}


@app.post("/api/rag/ingest")
async def rag_ingest(nights: int = 14) -> Dict[str, Any]:
    """Push period summaries into the knowledge graph."""
    rag = get_rag()
    span = store.span()
    if span is None:
        raise HTTPException(status_code=400, detail="No biometric data to ingest.")

    first = span[0].replace(hour=0, minute=0, second=0, microsecond=0)
    available = int((span[1] - first).total_seconds() // 86400) + 1
    windows = S.nightly_windows(first, min(nights, available))

    baseline: List[float] = []
    summaries = []
    for pid, start, end in windows:
        summ = S.summarize_period(
            store, start, end, pid, user=config.user,
            rmssd_baseline=baseline[-config.baseline_nights:],
        )
        if summ.metrics.n_beats:
            baseline.append(summ.metrics.rmssd_ms)
            summaries.append(summ)
    return {"ingested": await rag.ingest_summaries(summaries)}


@app.post("/api/query")
async def query(payload: QueryPayload) -> JSONResponse:
    rag = get_rag()
    answer = await rag.query(payload.question, mode=payload.mode)
    return JSONResponse({"question": payload.question, "answer": answer})


def main() -> None:  # pragma: no cover - entry point
    import uvicorn

    uvicorn.run(app, host=config.api_host, port=config.api_port)


if __name__ == "__main__":  # pragma: no cover
    main()
