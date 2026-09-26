"""Prompt templates for the biosignal query layer.

These deliberately live in a module-local dictionary rather than in
``raganything.prompt.PROMPTS``, for two concrete reasons:

1. ``prompt_manager.set_prompt_language()`` and ``reset_prompts()`` rebuild the
   global registry by iterating a snapshot of the English prompts taken at
   *import* time and swapping the whole dict. Any key added afterwards by an
   external module is silently erased the first time a caller switches
   language -- a failure that would surface as a ``KeyError`` in production,
   far from its cause.
2. ``tests/test_prompt_language.py::test_chinese_prompts_have_all_keys``
   asserts every English key has a Chinese counterpart. Adding a key to the
   global registry without translating it breaks an unrelated test.

The answer template is validated at import, because the one way to get it
catastrophically wrong -- omitting ``{context_data}`` -- fails *silently*: the
query still runs, the model still answers, and every piece of retrieved
evidence has been dropped on the floor.
"""

from __future__ import annotations

from typing import Dict

__all__ = ["BIOSIGNAL_PROMPTS", "SYSTEM_PROMPT_FIELDS"]

#: The only fields the underlying pipeline formats a caller-supplied system
#: prompt with. Any other brace in the template raises ``KeyError`` at query
#: time.
SYSTEM_PROMPT_FIELDS = ("response_type", "user_prompt", "context_data")

BIOSIGNAL_PROMPTS: Dict[str, str] = {}

BIOSIGNAL_PROMPTS[
    "ANSWER_SYSTEM_PROMPT"
] = """You are answering questions about one person's own physiological data, recorded from their fitness and health devices.

The context below comes from an ingestion system that records, for every measurement, how it was obtained and how much of the recording window was actually observed. It distinguishes four kinds of value:

- measured: a sensor reading, transported without reinterpretation
- locally_derived: computed from measured inputs by auditable code
- vendor_derived: produced by a closed algorithm that cannot be reproduced
- imputed: invented to cover a gap

Rules you must follow:

1. A metric marked "withheld" has NO value. Do not state one, do not estimate one, and do not infer one from a related metric. Say it was withheld and give the recorded reason.
2. Never present a vendor_derived score as if it were a measurement.
3. If the context reports gaps, lost sensor contact, or a low quality score for a stream, any conclusion drawn from that stream must carry that qualification in the same sentence.
4. State every number in the form `metric = value unit` so it can be checked against the underlying computation.
5. If the context does not contain what the question asks for, say so. Do not fill the gap from general knowledge about physiology.
6. This is analysis of consumer device data, not medical advice, and must never be phrased as diagnosis.

---Context---

{context_data}

---Response Rules---

Format the response as: {response_type}

{user_prompt}"""

BIOSIGNAL_PROMPTS["SCOPE_USER_PROMPT"] = (
    """Answer only from sessions between {start} and {end}. If the context contains no session in that range, say so plainly rather than answering from sessions outside it. Name the session id or date behind every claim you make."""
)

BIOSIGNAL_PROMPTS["CANONICAL_NUMBER_INSTRUCTION"] = (
    """State every numeric claim in the exact form `metric = value unit` (for example `hrv_rmssd = 42.1 ms`), using the metric names as they appear in the context."""
)

BIOSIGNAL_PROMPTS["DETERMINISTIC_FACTS_HEADER"] = (
    """The following figures were computed directly from the stored session records, not retrieved from text. They are exact. Use them verbatim; do not recompute, round differently, or contradict them. Your task is to explain and contextualise them, not to produce numbers of your own."""
)

BIOSIGNAL_PROMPTS[
    "REGENERATE_USER_PROMPT"
] = """Your previous answer made claims the underlying data does not support:

{violations}

Rewrite the answer without those claims. Where a metric was withheld, say it was withheld and give the reason; do not substitute a value. Keep everything the data does support."""

BIOSIGNAL_PROMPTS[
    "ROUTER_CLASSIFY"
] = """Classify this question about a person's physiological data.

Question: {question}

Reply with strict JSON and nothing else:

{{"route": one of ["deterministic", "retrieval", "hybrid"],
  "intent": one of ["aggregate", "trend", "compare", "lookup", "explain", "relate"],
  "metrics": a list drawn only from [{metrics}],
  "statistic": one of ["mean", "median", "min", "max", "sum", "count", "stdev", "first", "last"] or null}}

Use "deterministic" when the question asks for a number, an average, or a trend.
Use "retrieval" when it asks why something happened, or how two things relate.
Use "hybrid" when it asks for both, or when you are unsure.
Never invent a metric name that is not in the list."""


def _assert_format_fields(name: str, template: str) -> None:
    """Fail at import if a system-prompt template cannot be formatted.

    Catching this here rather than at query time matters because the specific
    failure of omitting ``{context_data}`` produces no error at all -- just a
    confident answer with no evidence behind it.
    """
    sentinel = "__CONTEXT_SENTINEL__"
    try:
        rendered = template.format(
            response_type="Multiple Paragraphs",
            user_prompt="",
            context_data=sentinel,
        )
    except (KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(
            f"{name} is not a valid system-prompt template: it may only reference "
            f"{SYSTEM_PROMPT_FIELDS}, and literal braces must be doubled ({exc})"
        ) from exc
    if sentinel not in rendered:
        raise RuntimeError(
            f"{name} drops {{context_data}}; a query using it would silently "
            "discard every piece of retrieved evidence"
        )


_assert_format_fields("ANSWER_SYSTEM_PROMPT", BIOSIGNAL_PROMPTS["ANSWER_SYSTEM_PROMPT"])
