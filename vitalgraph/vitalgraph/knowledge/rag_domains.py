"""Per-domain RAG: one knowledge graph per decision domain.

Each domain gets its own RAGAnything instance with its own working directory,
so twenty separate graphs exist rather than one pooled corpus. This is the
whole point of the partition: apnea literature and glycaemic literature share
almost no vocabulary, and a single graph forces them to compete for context
space on every query. Separate graphs also mean a domain's corpus can be
rebuilt, versioned or dropped without disturbing the other nineteen.

Each domain's graph is *seeded* from the structured knowledge before any
external literature arrives, so it can answer "what does this decision
require, and can our sensors deliver it?" on day one. Literature ingestion
then deepens a corpus that is already correct rather than filling an empty one.

``raganything`` is imported lazily, as in ``rag.py``: the knowledge base and
the routing must stay usable and testable without LightRAG, MinerU or an LLM
key present.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Tuple

from . import domains as domain_catalogue
from . import sensors as sensor_catalogue
from . import signals as signal_catalogue
from .domains import Domain

#: Root under which each domain gets its own storage directory.
DEFAULT_ROOT = "./vybe_rag"

#: Words too common or too content-free to discriminate between domains.
#: Function words need listing even when rare: "getting" appears in exactly one
#: domain, so inverse-document-frequency scores it as highly discriminative,
#: and "am I getting sick" routed to environmental exposure on the strength of
#: a gerund. Rarity is not the same as meaning.
_STOPWORDS = frozenset(
    """a an and are as at be by can do does for from have how i if in is it my
    of on or should that the to was what when where which who why will with
    you your am me being been over under about into more most this these those
    get getting got give gives got had has having just know let like make makes
    much many need needs really see tell than then there thing things too very
    want wants way well would could also any bad good right now today
    """.split()
)

#: Colloquial terms mapped to the catalogue's vocabulary. Users ask "am I
#: getting sick", not "is this pre-symptomatic infection", and routing on the
#: clinical phrasing alone strands the questions people actually type.
LAY_SYNONYMS: Dict[str, str] = {
    "sick": "illness infection detection",
    "ill": "illness infection",
    "illness": "illness infection",
    "unwell": "illness infection",
    "fever": "illness infection temperature",
    "cold": "illness infection",
    "flu": "illness infection",
    "tired": "readiness recovery training",
    "exhausted": "readiness recovery training",
    "fatigued": "readiness recovery training",
    "drained": "readiness recovery training",
    "hungover": "alcohol medication",
    "drinking": "alcohol medication",
    "drunk": "alcohol medication",
    "booze": "alcohol medication",
    "snoring": "snoring breathing apnea",
    "apnoea": "apnea breathing",
    "afib": "afib rhythm fibrillation",
    "palpitations": "rhythm irregular ectopic",
    "stressed": "stress autonomic",
    "anxious": "stress autonomic mood",
    "depressed": "mood behavioural",
    "period": "menstrual cycle phase",
    "ovulating": "ovulation menstrual cycle",
    "sugar": "glucose glycemic metabolic",
    "carbs": "glucose glycemic meal",
    "weight": "mass composition body",
    "sunlight": "light exposure circadian",
    "jetlag": "circadian alignment light",
    "altitude": "altitude oxygen saturation",
    "overtraining": "overreaching load training",
    "sore": "musculoskeletal load impact",
    # "VO2 max" tokenises to two words while the signal id is one, so the
    # question a user actually types matches nothing without this.
    "vo2": "vo2max aerobic fitness cardiorespiratory",
    "aerobic": "vo2max aerobic fitness cardiorespiratory",
    "cardio": "vo2max aerobic fitness cardiorespiratory",
    "vigilance": "alertness cognitive performance",
    "focus": "alertness cognitive performance",
    "alert": "alertness cognitive performance",
}


def working_dir_for(domain_id: str, root: str | Path = DEFAULT_ROOT) -> Path:
    """Storage directory for one domain's graph."""
    return Path(root) / domain_id


# --------------------------------------------------------------------------
# Seed content: the structured knowledge, per domain
# --------------------------------------------------------------------------


def _decision_rows(domain: Domain) -> str:
    rows = [
        "| Decision | Question | Requires | History | Evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for d in domain.decisions:
        history = f"{d.min_history_days}d" if d.min_history_days else "none"
        rows.append(
            f"| {d.id} | {d.question} | {', '.join(d.requires)} "
            f"| {history} | {d.evidence.value} |"
        )
    return "\n".join(rows)


def _adequacy_rows(domain: Domain) -> str:
    """Which sensors can deliver this domain's required signals, adequately.

    The most useful single table in the corpus: it answers, per domain, whether
    a given hardware stack can reach the decisions at all.
    """
    rows = [
        "| Required signal | Min rate | Adequate sensors | Inadequate sensors |",
        "| --- | --- | --- | --- |",
    ]
    for signal_id in domain.required_roots:
        signal = signal_catalogue.get(signal_id)
        providers = sensor_catalogue.providers_of(signal_id)
        adequate = [s.id for s in providers if s.meets_minimum(signal_id)]
        inadequate = [s.id for s in providers if s.meets_minimum(signal_id) is False]
        rate = f"{signal.min_sampling_hz:g} Hz" if signal.is_continuous else "episodic"
        rows.append(
            f"| {signal_id} | {rate} | {', '.join(adequate) or 'none'} "
            f"| {', '.join(inadequate) or 'none'} |"
        )
    return "\n".join(rows)


def seed_content_list(domain: Domain) -> List[Dict[str, Any]]:
    """Structured knowledge for one domain, as insertable content.

    Deterministically rendered from the catalogues, so re-seeding updates
    rather than duplicating -- the same discipline as the biometric summariser.
    """
    narrative = [
        f"Decision domain: {domain.name} (id {domain.id}, "
        f"RAG namespace {domain.namespace}).",
        domain.summary,
        f"This domain covers {len(domain.decisions)} decision(s): "
        + "; ".join(d.question for d in domain.decisions),
        f"Signals required across the domain: {', '.join(domain.required_signals)}.",
        f"Resolved to measured signals, the sensor requirement is: "
        f"{', '.join(domain.required_roots)}.",
        f"Strongest evidence grade among its decisions: {domain.best_evidence.value}.",
    ]
    if domain.corpus_topics:
        narrative.append(
            "Literature topics belonging to this corpus: "
            + ", ".join(domain.corpus_topics)
            + "."
        )
    if domain.notes:
        narrative.append(f"Note: {domain.notes}")

    for d in domain.decisions:
        if d.notes:
            narrative.append(f"On {d.id} ({d.evidence.value} evidence): {d.notes}")

    return [
        {"type": "text", "text": "\n".join(narrative), "page_idx": 0},
        {
            "type": "table",
            "table_body": _decision_rows(domain),
            "table_caption": [f"Decisions in {domain.name}"],
            "table_footnote": [
                "Evidence grades describe the supporting literature, not the "
                "confidence of any particular output: a high-grade decision "
                "computed from an inadequate sensor is still unsupported."
            ],
            "page_idx": 0,
        },
        {
            "type": "table",
            "table_body": _adequacy_rows(domain),
            "table_caption": [f"Sensor adequacy for {domain.name}"],
            "table_footnote": [
                "A sensor listed as inadequate delivers the signal below the "
                "rate the inference requires. That is different from not "
                "delivering it at all, and calls for a different fix."
            ],
            "page_idx": 0,
        },
    ]


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------


def _terms(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS}


#: Where a term appears matters as much as whether it does. "recovery" in a
#: domain's own name is a stronger signal than "recovery" inside one decision
#: id three domains away -- without this, the two tie and the winner is decided
#: alphabetically, which is how "why was my recovery bad" missed training
#: readiness.
_FIELD_WEIGHTS = {"name": 4.0, "summary": 2.5, "decision": 1.5, "topic": 1.0}


def _domain_fields(domain: Domain) -> Dict[str, str]:
    return {
        "name": " ".join((domain.id.replace("_", " "), domain.name)),
        "summary": domain.summary,
        "decision": " ".join(
            [d.question for d in domain.decisions]
            + [d.id.replace("_", " ") for d in domain.decisions]
            + [s.replace("_", " ") for d in domain.decisions for s in d.requires]
        ),
        "topic": " ".join(domain.corpus_topics),
    }


def _domain_vocabulary(domain: Domain) -> Dict[str, float]:
    """Term -> field weight, keeping the strongest field a term appears in."""
    out: Dict[str, float] = {}
    for field_name, text in _domain_fields(domain).items():
        weight = _FIELD_WEIGHTS[field_name]
        for term in _terms(text):
            if out.get(term, 0.0) < weight:
                out[term] = weight
    return out


#: Precomputed once; the catalogues are static.
_VOCABULARY: Dict[str, Dict[str, float]] = {
    d.id: _domain_vocabulary(d) for d in domain_catalogue.DOMAINS
}


def _build_weights() -> Dict[str, float]:
    """Inverse-document-frequency weight per vocabulary term.

    Without this, routing is dominated by words shared across many domains.
    "sleep" appears in most of them and discriminates almost nothing;
    "alcohol" and "glucose" each appear in one and are decisive. Counting raw
    term overlap treats them alike, which sent "does alcohol wreck my sleep"
    to blood pressure on an alphabetical tie-break.
    """
    total = len(_VOCABULARY) or 1
    frequency: Dict[str, int] = {}
    for vocabulary in _VOCABULARY.values():
        for term in vocabulary:  # keys only; field weight is applied separately
            frequency[term] = frequency.get(term, 0) + 1
    # +1 inside the log keeps a term present in every domain at a small
    # positive weight rather than exactly zero.
    return {term: math.log(1.0 + total / count) for term, count in frequency.items()}


_WEIGHTS: Dict[str, float] = _build_weights()


def term_weight(term: str) -> float:
    """How discriminative a term is. Unseen terms carry no weight."""
    return _WEIGHTS.get(term, 0.0)


def expand_terms(terms: Iterable[str]) -> set[str]:
    """Add catalogue vocabulary implied by colloquial words."""
    out = set(terms)
    for term in list(out):
        replacement = LAY_SYNONYMS.get(term)
        if replacement:
            out.update(_terms(replacement))
    return out


@dataclass(frozen=True, slots=True)
class RouteMatch:
    domain_id: str
    score: float
    matched: Tuple[str, ...]


def route(question: str, limit: int = 3) -> List[RouteMatch]:
    """Pick the domains a free-text question belongs to.

    Deliberately lexical rather than model-based: routing must be deterministic
    and free, must work with no LLM key, and must be inspectable when it picks
    wrongly. Returns several candidates because real questions genuinely span
    domains -- "why is my recovery bad after drinking" is both readiness and
    medication response.
    """
    asked = expand_terms(_terms(question))
    if not asked:
        return []

    # Normalise by the weight of the question's terms that appear anywhere in
    # the catalogue, so an unrecognised word does not depress every score.
    total = sum(term_weight(t) * max(_FIELD_WEIGHTS.values()) for t in asked)
    if total <= 0.0:
        return []

    matches: List[RouteMatch] = []
    for domain_id, vocabulary in _VOCABULARY.items():
        overlap = asked & set(vocabulary)
        if not overlap:
            continue
        weight = sum(term_weight(t) * vocabulary[t] for t in overlap)
        matches.append(
            RouteMatch(
                domain_id=domain_id,
                score=weight / total,
                # Most discriminative term first: this is the explanation of
                # why the question routed here.
                matched=tuple(sorted(overlap, key=lambda t: -term_weight(t))),
            )
        )
    matches.sort(key=lambda m: (-m.score, m.domain_id))
    return matches[:limit]


# --------------------------------------------------------------------------
# Per-domain RAG instances
# --------------------------------------------------------------------------

#: System prompt applied to every domain query. Scoped to the domain so an
#: answer cannot quietly wander into a neighbouring one.
DOMAIN_PROMPT = """You are answering within the Vybe decision domain "{name}".

{summary}

Rules:
- Answer only within this domain. If the question belongs to another domain,
  say which one rather than guessing across the boundary.
- Ground every claim in the retrieved context and cite its source.
- A decision is supportable only if its required signals are present AND
  sampled at or above their stated minimum rate. State when they are not.
- Distinguish measured values from derived ones and from model estimates.
- Report the evidence grade behind a claim when the context provides it.
"""


class DomainRAGUnavailable(RuntimeError):
    """Raised when per-domain RAG is used without ``raganything`` installed."""


@dataclass
class DomainRAG:
    """One domain's knowledge graph."""

    domain: Domain
    rag: Any

    async def seed(self) -> Dict[str, Any]:
        """Insert this domain's structured knowledge.

        Uses a deterministic doc id so re-seeding replaces rather than
        accumulating near-duplicate copies of the same catalogue.
        """
        content = seed_content_list(self.domain)
        await self.rag.insert_content_list(
            content_list=content,
            file_path=f"knowledge://{self.domain.namespace}",
            doc_id=f"vybe-domain-{self.domain.id}",
        )
        return {
            "domain": self.domain.id,
            "namespace": self.domain.namespace,
            "items": len(content),
        }

    async def query(self, question: str, mode: str = "mix") -> str:
        prompt = DOMAIN_PROMPT.format(
            name=self.domain.name, summary=self.domain.summary
        )
        return await self.rag.aquery(question, mode=mode, system_prompt=prompt)


class DomainRAGRegistry:
    """Lazily builds and holds one RAG per domain.

    Instances are created on demand: standing up twenty LightRAG graphs when
    only one is needed would be wasteful, and most sessions touch a handful.
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_ROOT,
        factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self._factory = factory or self._default_factory
        self._instances: Dict[str, DomainRAG] = {}

    @staticmethod
    def _default_factory(working_dir: str) -> Any:
        try:
            from ..rag import build_openai_rag
        except ImportError as exc:  # pragma: no cover - defensive
            raise DomainRAGUnavailable(str(exc)) from exc
        return build_openai_rag(working_dir)

    def get(self, domain_id: str) -> DomainRAG:
        if domain_id not in self._instances:
            domain = domain_catalogue.get(domain_id)
            rag = self._factory(str(working_dir_for(domain.id, self.root)))
            self._instances[domain_id] = DomainRAG(domain=domain, rag=rag)
        return self._instances[domain_id]

    @property
    def loaded(self) -> List[str]:
        return sorted(self._instances)

    def layout(self) -> Dict[str, str]:
        """Where each domain's graph lives on disk."""
        return {
            d.id: str(working_dir_for(d.id, self.root))
            for d in domain_catalogue.DOMAINS
        }

    async def seed(
        self, domain_ids: Iterable[str] | None = None
    ) -> List[Dict[str, Any]]:
        """Seed the named domains, or all of them."""
        ids = (
            list(domain_ids)
            if domain_ids is not None
            else [d.id for d in domain_catalogue.DOMAINS]
        )
        return [await self.get(domain_id).seed() for domain_id in ids]

    async def query(
        self, domain_id: str, question: str, mode: str = "mix"
    ) -> Dict[str, Any]:
        rag = self.get(domain_id)
        answer = await rag.query(question, mode=mode)
        return {
            "domain": domain_id,
            "namespace": rag.domain.namespace,
            "question": question,
            "answer": answer,
        }

    async def query_routed(
        self, question: str, limit: int = 1, mode: str = "mix"
    ) -> List[Dict[str, Any]]:
        """Route a free-text question and query the best-matching domains."""
        matches = route(question, limit=limit)
        if not matches:
            return []
        out = []
        for match in matches:
            result = await self.query(match.domain_id, question, mode=mode)
            result["route_score"] = round(match.score, 4)
            result["route_matched"] = list(match.matched)
            out.append(result)
        return out


def corpus_plan() -> List[Dict[str, Any]]:
    """What each domain's corpus should contain -- the harvesting work list.

    Feeds the acquisition layer: every topic here is a search to run, and the
    results land in that domain's graph and nowhere else.
    """
    return [
        {
            "domain": d.id,
            "namespace": d.namespace,
            "working_dir": str(working_dir_for(d.id)),
            "topics": list(d.corpus_topics),
            "required_roots": d.required_roots,
            "decisions": len(d.decisions),
            "best_evidence": d.best_evidence.value,
        }
        for d in domain_catalogue.DOMAINS
    ]
