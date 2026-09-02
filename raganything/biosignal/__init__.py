"""Evidence-preserving ingestion and analysis of BLE fitness and health signals.

The subsystem has one organising principle: **never let an inference wear the
clothes of a measurement.** Signals enter through :mod:`.ble` (raw GATT),
:mod:`.sources` (vendor clouds and phone health stores), or a manual entry, and
every value carries how it was obtained, how much of the window it actually
covered, and how much to trust it. :mod:`.analytics` refuses to emit metrics
their inputs cannot support, and :mod:`.narrative` renders sessions -- caveats
included -- into RAG-Anything's knowledge graph, where physiology can finally be
queried alongside training plans, lab results and literature.

See ``docs/ble_fitness_gap_analysis.md`` for what today's platforms discard and
why each of those omissions is addressed here.
"""

from . import (
    analytics,
    ble,
    index,
    narrative,
    prompts,
    query,
    router,
    sources,
    store,
    timeseries,
    verify,
)
from .narrative import SessionReport, analyze_session, to_content_list
from .query import BiosignalAnswer, BiosignalQueryEngine, aquery_biosignal
from .router import QueryPlan, Route
from .store import ReportRecord, ReportStore
from .schema import (
    Evidence,
    Modality,
    Provenance,
    Sample,
    Session,
    SourceKind,
    Stream,
    make_stream,
)

__all__ = [
    "analytics",
    "ble",
    "index",
    "narrative",
    "prompts",
    "query",
    "router",
    "sources",
    "store",
    "timeseries",
    "verify",
    "Evidence",
    "Modality",
    "Provenance",
    "Sample",
    "Session",
    "SessionReport",
    "SourceKind",
    "Stream",
    "analyze_session",
    "aquery_biosignal",
    "make_stream",
    "to_content_list",
    "BiosignalAnswer",
    "BiosignalQueryEngine",
    "QueryPlan",
    "ReportRecord",
    "ReportStore",
    "Route",
]
