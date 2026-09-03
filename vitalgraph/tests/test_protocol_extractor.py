"""Mining BLE protocol facts from source.

These facts are what survives the license gate from every source, so the
extractor is the load-bearing piece of the reverse-engineering capability.
"""

from vitalgraph.protocol.extractor import ProtocolRegistry, extract_facts

VENDOR_SRC = """
HEART_RATE_CHAR = "0x2A37"

def parse(data):
    flags = data[0]
    if flags & 0x10:
        return struct.unpack_from("<H", data, 1)
"""

PROPRIETARY_SRC = """
VENDOR_SERVICE = "6e400001-b5a3-f393-e0a9-e50e24dcca9e"
SIG_IN_DISGUISE = "0000180d-0000-1000-8000-00805f9b34fb"
"""


def test_extracts_16_bit_uuids():
    facts = extract_facts(VENDOR_SRC, "p.py")
    assert "0x2A37" in {f.uuid for f in facts}


def test_recovers_wire_format_from_code():
    """The capability that matters: byte layout learned from an implementation."""
    facts = extract_facts(VENDOR_SRC, "p.py")
    kinds = {f.kind for f in facts}
    assert {"byte_offset", "bit_mask", "struct_format"} <= kinds

    details = {f.detail for f in facts}
    assert "reads byte 0" in details
    assert "tests bit mask 0x10" in details  # the RR-present flag
    assert "struct format <H" in details  # little-endian uint16


def test_vendor_128_bit_uuids_are_flagged_proprietary():
    facts = extract_facts(PROPRIETARY_SRC, "v.py")
    vendor = [f for f in facts if f.kind == "uuid_128"]
    assert vendor and vendor[0].uuid.startswith("6e400001")


def test_sig_uuid_written_as_128_bit_is_normalised():
    """A SIG UUID in 128-bit form must not be reported as proprietary."""
    facts = extract_facts(PROPRIETARY_SRC, "v.py")
    assert "0x180D" in {f.uuid for f in facts}
    assert not any(f.uuid.startswith("0000180d") for f in facts)


def test_common_noise_values_are_not_treated_as_uuids():
    facts = extract_facts("mask = 0xFFFF\nzero = 0x0000\n", "n.py")
    assert not [f for f in facts if f.kind == "uuid_16"]


def test_citations_point_at_the_line():
    facts = extract_facts(VENDOR_SRC, "ble/p.py", repo="org/repo", ref="abc123")
    f = next(f for f in facts if f.uuid == "0x2A37")
    assert f.citation() == f"org/repo@abc123:ble/p.py#L{f.line}"


def test_registry_counts_and_groups_evidence():
    reg = ProtocolRegistry()
    reg.add(extract_facts(VENDOR_SRC, "p.py", repo="a"))
    assert reg.evidence_count()["0x2A37"] > 1
    assert "0x2A37" in reg.uuids()
    assert reg.by_uuid("0x2a37")  # case-insensitive lookup


def test_corroboration_requires_independent_repos():
    """One project's quirk is not a protocol fact; two agreeing is evidence."""
    reg = ProtocolRegistry()
    reg.add(extract_facts(VENDOR_SRC, "p.py", repo="vendor/a"))
    assert reg.corroborated() == []

    reg.add(extract_facts(VENDOR_SRC, "q.py", repo="vendor/b"))
    assert ("0x2A37", 2) in reg.corroborated()


def test_registry_bridges_uuids_to_derivable_signals():
    """'We found this UUID' becomes 'this device can measure X'."""
    reg = ProtocolRegistry()
    reg.add(extract_facts(VENDOR_SRC, "p.py"))
    derivable = reg.derivable_for(reg.uuids())
    assert "rmssd" in derivable["0x2A37"]


def test_content_list_shape_and_caveat():
    reg = ProtocolRegistry()
    reg.add(extract_facts(VENDOR_SRC, "p.py"))
    items = reg.to_content_list()
    assert items[0]["type"] == "table"
    assert "0x2A37" in items[0]["table_body"]
    # The locality heuristic must be disclosed, not hidden.
    assert "heuristic" in items[0]["table_footnote"][0].lower()


def test_empty_registry_emits_nothing():
    assert ProtocolRegistry().to_content_list() == []


def test_uuid_normalisation_is_canonical():
    """Regression: `uuid.upper()` yields 0X2A37 and silently misses a registry
    keyed as 0x2A37, returning an empty result instead of an error."""
    from vitalgraph.protocol.extractor import normalise_uuid

    for variant in ("0x2A37", "0X2a37", "2a37", " 0x2a37 "):
        assert normalise_uuid(variant) == "0x2A37"

    vendor = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
    assert normalise_uuid(vendor) == vendor.lower()


def test_lookups_are_case_insensitive_end_to_end():
    reg = ProtocolRegistry()
    reg.add(extract_facts(VENDOR_SRC, "p.py"))
    for variant in ("0x2A37", "0X2a37", "2A37"):
        assert reg.by_uuid(variant), f"lookup failed for {variant}"
        assert "rmssd" in reg.derivable_for([variant])["0x2A37"]
