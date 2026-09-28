"""Rendering the device catalogue as knowledge-graph content."""

from vitalgraph.devices.catalog import DEVICES
from vitalgraph.devices.content import device_landscape_content_list


def test_content_list_shape_matches_insert_content_list_contract():
    items = device_landscape_content_list()
    assert [i["type"] for i in items] == ["text", "table"]
    assert items[0]["text"].strip()
    assert items[1]["table_body"].startswith("| Device |")


def test_every_device_is_named_in_the_table():
    body = device_landscape_content_list()[1]["table_body"]
    for d in DEVICES:
        assert d.name in body


def test_narrative_states_the_reachability_split():
    text = device_landscape_content_list()[0]["text"]
    assert "open" in text.lower()
    assert "cloud" in text.lower() or "oauth" in text.lower()


def test_narrative_makes_the_central_point_about_cloud_apis():
    """The whole reason this content exists: an API does not upgrade the
    sensor behind it."""
    text = device_landscape_content_list()[0]["text"]
    assert "does not" in text.lower() or "not upgraded" in text.lower()


def test_footnote_distinguishes_the_three_access_modes():
    footnote = device_landscape_content_list()[1]["table_footnote"][0]
    assert "OPEN_BLE" in footnote or "open" in footnote.lower()
    assert "CLOUD_API" in footnote or "cloud" in footnote.lower()
