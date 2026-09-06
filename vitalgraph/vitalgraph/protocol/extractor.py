"""Mining BLE protocol facts out of source code.

This is the part of ingestion that survives the license gate regardless of
source. A UUID, a characteristic's byte layout, the scaling applied to a field
-- these are interoperability facts about how a device behaves on the wire, not
creative expression. They can be learned from any implementation, including one
whose source may never be reproduced.

Output is deliberately structured rather than prose: it is the seed of the
queryable protocol registry, whose job is to answer "what can this hardware
actually measure?" mechanically instead of by retrieval guesswork.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple

from ..ble.gatt import DERIVABLE_FROM

#: 16-bit GATT UUIDs, written as `0x2A37`, `2A37`, or inside a 128-bit base.
_UUID16 = re.compile(r"\b0[xX]([0-9a-fA-F]{4})\b")

#: Full 128-bit UUIDs, which is how vendors express proprietary services.
_UUID128 = re.compile(
    r"\b([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})\b"
)

#: The Bluetooth SIG base UUID. A 128-bit UUID matching it is really a 16-bit
#: standard UUID in disguise, and must not be reported as proprietary.
_SIG_BASE_SUFFIX = "-0000-1000-8000-00805f9b34fb"

#: Python struct format strings, which encode endianness and field widths.
_STRUCT_FMT = re.compile(r"""["']([<>!@=][BbHhIiLlQqfd]+)["']""")

#: Byte-offset indexing, e.g. `data[3]` or `payload[1:3]`.
_BYTE_INDEX = re.compile(r"\b(?:data|payload|buf|buffer|bytes|value)\s*\[\s*(\d+)")

#: Bit-mask tests, which is how flag bytes are decoded.
_BIT_MASK = re.compile(r"&\s*0[xX]([0-9a-fA-F]{1,2})\b")

#: UUID-shaped hex that is almost always something else.
_UUID16_NOISE = {"0000", "FFFF", "00FF", "FF00", "0001", "1000", "8000"}

#: Ranges the Bluetooth SIG actually allocates 16-bit UUIDs from, as inclusive
#: ``(low, high, what)`` triples.
#:
#: The first harvest is what forced this. Any four-digit hex literal looks like
#: a 16-bit UUID, so a bare pattern match reported ``0x1021`` -- the CCITT
#: CRC-16 polynomial -- as a characteristic, along with array bounds, bit masks
#: and version constants: roughly nine tenths of the 3620 "protocol facts" from
#: ten repositories were nothing of the kind. A blocklist cannot fix that,
#: because the noise is unbounded and the signal is not. The allocated ranges
#: are, so this is an allowlist.
SIG_UUID16_RANGES: Tuple[Tuple[int, int, str], ...] = (
    (0x1800, 0x18FF, "GATT service"),
    (0x2700, 0x27FF, "unit"),
    (0x2800, 0x28FF, "attribute type declaration"),
    (0x2900, 0x29FF, "characteristic descriptor"),
    (0x2A00, 0x2BFF, "characteristic type"),
    (0xFC00, 0xFDFF, "member service (vendor-allocated by the SIG)"),
    (0xFE00, 0xFEFF, "member service (vendor-allocated by the SIG)"),
)

#: Words that make a line about UUIDs regardless of the value's range. This is
#: how a genuinely proprietary 16-bit UUID outside SIG ranges is still caught:
#: the code around it says what it is.
#:
#: Boundaries are ``(?<![A-Za-z0-9])`` rather than ``\b`` because identifiers
#: are written ``VENDOR_CHAR_UUID`` and ``service_uuid``: underscore is a word
#: character, so ``\b`` refuses to match at exactly the places these words
#: appear in code.
_UUID_CONTEXT = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(uuid|guid|characteristic|service|descriptor|gatt|assigned_number)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


#: Evidence that a file concerns Bluetooth at all. Checked once per file, not
#: per line, because a module's subject is a property of the module.
#: Same underscore-aware boundaries as ``_UUID_CONTEXT``: ``ble_device`` and
#: ``gatt_server`` are exactly the names this has to catch.
_BLUETOOTH_FILE_CONTEXT = re.compile(
    r"(?<![A-Za-z0-9])"
    # Deliberately narrow. "characteristic" and "peripheral" were here and had
    # to go: FHIR uses `characteristic` as a field name throughout
    # PlanDefinition and Group, and "peripheral" is ordinary clinical English.
    # Both let a FHIR client's test fixtures register as GATT services. A
    # marker earns its place only if it means Bluetooth and nothing else.
    r"(bluetooth|bluez|corebluetooth|gatt|btle|ble|bleak|l2cap)"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)


def is_bluetooth_context(source: str, path: str) -> bool:
    """Whether a file is plausibly about Bluetooth.

    A 128-bit UUID means "proprietary GATT service" only in a file that has
    something to do with Bluetooth. Elsewhere it is a primary key or a test
    fixture, and reporting it as a vendor's service is a fabricated protocol
    fact -- the exact failure this registry must not produce.
    """
    if _BLUETOOTH_FILE_CONTEXT.search(path):
        return True
    return bool(_BLUETOOTH_FILE_CONTEXT.search(source))


def sig_uuid16_role(value: int) -> str | None:
    """What the SIG allocates ``value`` for, or None if it allocates nothing."""
    for low, high, role in SIG_UUID16_RANGES:
        if low <= value <= high:
            return role
    return None


def is_plausible_uuid16(raw: str, line: str) -> bool:
    """Whether a four-digit hex literal is plausibly a 16-bit UUID.

    Accepts a value inside a SIG-allocated range, or any value on a line whose
    wording is about UUIDs. Rejecting everything else is what keeps a CRC
    polynomial out of the protocol registry. This under-reports rather than
    over-reports on purpose: a missing fact is a gap, while a fabricated one is
    a confidently wrong answer about how a device works.
    """
    upper = raw.upper()
    if upper in _UUID16_NOISE:
        return False
    if sig_uuid16_role(int(upper, 16)) is not None:
        return True
    return bool(_UUID_CONTEXT.search(line))


@dataclass(frozen=True, slots=True)
class ProtocolFact:
    """One behavioural observation about a characteristic or service."""

    uuid: str
    kind: str
    """uuid_16 | uuid_128 | struct_format | byte_offset | bit_mask"""
    detail: str
    source_path: str
    line: int
    repo: str = "local"
    ref: str | None = None

    def citation(self) -> str:
        prefix = f"{self.repo}@{self.ref}" if self.ref else self.repo
        return f"{prefix}:{self.source_path}#L{self.line}"


def normalise_uuid(uuid: str) -> str:
    """Canonical form for any UUID: ``0xABCD`` for 16-bit, lowercase for 128-bit.

    Every lookup goes through this. Ad-hoc casing at each call site is how
    ``uuid.upper()`` silently stops matching a registry keyed as ``0x2A37`` --
    a mismatch that returns an empty result rather than an error.
    """
    stripped = uuid.strip()
    if stripped.lower().startswith("0x"):
        return f"0x{stripped[2:].upper()}"
    if len(stripped) == 4 and all(c in "0123456789abcdefABCDEF" for c in stripped):
        return f"0x{stripped.upper()}"
    return stripped.lower()


def _normalise_uuid16(raw: str) -> str:
    return f"0x{raw.upper()}"


def _is_sig_uuid(uuid128: str) -> str | None:
    """If a 128-bit UUID is a SIG UUID in disguise, return its 16-bit form."""
    lowered = uuid128.lower()
    if lowered.endswith(_SIG_BASE_SUFFIX) and lowered.startswith("0000"):
        return f"0x{lowered[4:8].upper()}"
    return None


def extract_facts(
    source: str, path: str, repo: str = "local", ref: str | None = None
) -> List[ProtocolFact]:
    """Extract protocol facts from one source file.

    Scans for UUIDs and, near each, the decoding machinery that reveals a
    characteristic's wire format: struct formats, byte offsets and bit masks.
    Locality is the heuristic -- decoding code sits close to the UUID it
    decodes -- so results are leads to verify, not established truth.
    """
    facts: List[ProtocolFact] = []
    lines = source.splitlines()
    bluetooth_file = is_bluetooth_context(source, path)

    for lineno, line in enumerate(lines, start=1):
        for match in _UUID128.finditer(line):
            raw = match.group(1)
            sig_form = _is_sig_uuid(raw)
            if not sig_form and not bluetooth_file:
                # A 128-bit UUID in a file that is not about Bluetooth is a
                # database key, a test fixture or a namespace id -- anything
                # but a proprietary GATT service. Calling it one was how a
                # FHIR client's `urn:uuid:` fixtures entered the registry.
                #
                # File context, not line context: FHIR's own `urn:uuid:...`
                # contains the word "uuid", so the line says nothing useful.
                # What a UUID means is a property of the module it lives in.
                continue
            if sig_form:
                facts.append(
                    ProtocolFact(
                        sig_form,
                        "uuid_16",
                        f"SIG UUID written in 128-bit form ({raw})",
                        path,
                        lineno,
                        repo,
                        ref,
                    )
                )
            else:
                facts.append(
                    ProtocolFact(
                        raw.lower(),
                        "uuid_128",
                        "vendor-proprietary UUID",
                        path,
                        lineno,
                        repo,
                        ref,
                    )
                )

        for match in _UUID16.finditer(line):
            raw = match.group(1).upper()
            if not is_plausible_uuid16(raw, line):
                continue
            facts.append(
                ProtocolFact(
                    _normalise_uuid16(raw),
                    "uuid_16",
                    line.strip()[:120],
                    path,
                    lineno,
                    repo,
                    ref,
                )
            )

    # Decoding machinery, attributed to the nearest preceding UUID.
    current: str | None = None
    for lineno, line in enumerate(lines, start=1):
        uuid_here = _UUID128.search(line) or _UUID16.search(line)
        if uuid_here:
            raw = uuid_here.group(1)
            if len(raw) == 4 and is_plausible_uuid16(raw, line):
                current = _normalise_uuid16(raw)
            elif len(raw) > 4:
                sig_form = _is_sig_uuid(raw)
                # Same gate as the first pass. Without it, a 128-bit value
                # rejected as a UUID above still became the anchor that later
                # byte offsets were attributed to, so the decoding facts
                # survived even though the UUID they name did not.
                if sig_form or bluetooth_file:
                    current = sig_form or raw.lower()
        if current is None:
            continue
        for match in _STRUCT_FMT.finditer(line):
            facts.append(
                ProtocolFact(
                    current,
                    "struct_format",
                    f"struct format {match.group(1)}",
                    path,
                    lineno,
                    repo,
                    ref,
                )
            )
        for match in _BYTE_INDEX.finditer(line):
            facts.append(
                ProtocolFact(
                    current,
                    "byte_offset",
                    f"reads byte {match.group(1)}",
                    path,
                    lineno,
                    repo,
                    ref,
                )
            )
        for match in _BIT_MASK.finditer(line):
            facts.append(
                ProtocolFact(
                    current,
                    "bit_mask",
                    f"tests bit mask 0x{match.group(1).upper()}",
                    path,
                    lineno,
                    repo,
                    ref,
                )
            )

    return facts


@dataclass
class ProtocolRegistry:
    """Accumulated protocol facts, queryable by UUID."""

    facts: List[ProtocolFact] = field(default_factory=list)

    def add(self, facts: Iterable[ProtocolFact]) -> int:
        before = len(self.facts)
        self.facts.extend(facts)
        return len(self.facts) - before

    def uuids(self) -> List[str]:
        return sorted({f.uuid for f in self.facts})

    def by_uuid(self, uuid: str) -> List[ProtocolFact]:
        key = normalise_uuid(uuid)
        return [f for f in self.facts if normalise_uuid(f.uuid) == key]

    def evidence_count(self) -> Dict[str, int]:
        counts: Dict[str, int] = defaultdict(int)
        for f in self.facts:
            counts[f.uuid] += 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def corroborated(self, min_repos: int = 2) -> List[Tuple[str, int]]:
        """UUIDs observed in several independent repositories.

        Independent corroboration is the strongest signal a protocol fact is
        real rather than one project's idiosyncrasy.
        """
        repos: Dict[str, set[str]] = defaultdict(set)
        for f in self.facts:
            repos[f.uuid].add(f.repo)
        out = [(uuid, len(rs)) for uuid, rs in repos.items() if len(rs) >= min_repos]
        out.sort(key=lambda kv: (-kv[1], kv[0]))
        return out

    def derivable_for(self, uuids: Iterable[str]) -> Dict[str, List[str]]:
        """Health signals computable from the given characteristics.

        Bridges mined UUIDs to ``ble/gatt.py``'s ``DERIVABLE_FROM``, which is
        what turns "we found this UUID" into "this device can measure X".
        """
        out: Dict[str, List[str]] = {}
        for uuid in uuids:
            key = normalise_uuid(uuid)
            if key in DERIVABLE_FROM:
                out[key] = DERIVABLE_FROM[key]
        return out

    def to_content_list(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Render the registry as a table item for the knowledge graph."""
        counts = self.evidence_count()
        if not counts:
            return []
        rows = [
            "| UUID | Observations | Kinds | Example evidence |",
            "| --- | --- | --- | --- |",
        ]
        for uuid, count in list(counts.items())[:limit]:
            facts = self.by_uuid(uuid)
            kinds = ", ".join(sorted({f.kind for f in facts}))
            example = next(
                (f.detail for f in facts if f.kind != "uuid_16"), facts[0].detail
            )
            rows.append(f"| `{uuid}` | {count} | {kinds} | {example[:70]} |")

        return [
            {
                "type": "table",
                "table_body": "\n".join(rows),
                "table_caption": ["BLE protocol facts mined from ingested source"],
                "table_footnote": [
                    "Decoding details are attributed to the nearest preceding UUID, "
                    "a locality heuristic. Treat rows as leads to verify against the "
                    "Bluetooth SIG specification or the device itself."
                ],
                "page_idx": 0,
            }
        ]
