"""Tests for the capability report.

What is being protected here is not layout, it is honesty. A status page is the
one artifact a reader takes at face value, so the tests assert that the
unflattering facts survive rendering: the failing subjects, the factors nothing
can sense, and the licence policy that holds three repositories to facts only.

A page that dropped those would still render perfectly.
"""

from __future__ import annotations

import re

import pytest

from tools.status_page import build, count_tests, render, subject_chart
from vitalgraph.knowledge.factors import FIVE_FACTORS, Factor


@pytest.fixture(scope="module")
def page(tmp_path_factory) -> str:
    return build(tmp_path_factory.mktemp("docs") / "status.html").read_text()


# --- the page is built from the code, not typed out ------------------------


def test_every_decoded_characteristic_appears(page: str):
    """The sensor table is generated from the decoder constants, so a
    characteristic that exists in code cannot be missing from the page."""
    from vitalgraph.ble import measurements as m

    for uuid in (
        m.CHAR_BODY_COMPOSITION_MEASUREMENT,
        m.CHAR_WEIGHT_MEASUREMENT,
        m.CHAR_GLUCOSE_MEASUREMENT,
        m.CHAR_BLOOD_PRESSURE_MEASUREMENT,
        m.CHAR_RSC_MEASUREMENT,
        m.CHAR_CSC_MEASUREMENT,
    ):
        assert uuid in page


def test_all_five_factors_are_shown(page: str):
    for factor in Factor:
        assert factor.value.title() in page
        assert FIVE_FACTORS[factor].core_question in page


def test_the_test_count_is_counted_not_asserted():
    """Counted from the test files, so the number on the page cannot drift."""
    assert count_tests() > 400


# --- the unflattering facts survive ----------------------------------------


def test_the_worst_subject_is_named_on_the_page(page: str):
    """slp01a scores 9.2% against a 44.4% baseline. It is the single most
    important number here and it must not be averaged away."""
    assert "slp01a" in page
    assert "9.2%" in page


def test_the_number_of_subjects_without_skill_is_stated(page: str):
    assert "10 of 16 subjects show no" in page.replace("\n", " ").replace("  ", " ")


def test_connection_is_shown_as_having_no_passive_input(page: str):
    """The factor with no sensor behind it, deliberately. If this ever renders
    as an empty cell instead of a stated absence, the page has started to look
    like the gap is an oversight rather than the finding."""
    assert not FIVE_FACTORS[Factor.CONNECTION].passive_inputs
    assert "invisible to a band" in page


def test_the_not_built_section_is_present(page: str):
    for expected in ("Accelerometer input", "Multi-user isolation", "Anything clinical"):
        assert expected in page


def test_facts_only_repositories_are_labelled(page: str):
    """Three repositories contribute protocol facts and no source. If that
    label vanished, the page would imply we ingested copyleft code verbatim."""
    assert page.count("Facts only") >= 3


# --- the chart -------------------------------------------------------------


def test_the_chart_draws_one_bar_per_subject():
    per_subject = {
        "a": {"margin_over_baseline": 0.1, "has_skill": True},
        "b": {"margin_over_baseline": -0.2, "has_skill": False},
        "c": {"margin_over_baseline": 0.0, "has_skill": False},
    }
    svg = subject_chart(per_subject)
    assert svg.count("<rect") == 3
    assert svg.count("bar-noskill") == 2


def test_a_zero_margin_still_draws_a_visible_mark():
    """A subject exactly at its baseline must not vanish from the chart."""
    svg = subject_chart({"z": {"margin_over_baseline": 0.0, "has_skill": False}})
    width = float(re.search(r'<rect x="[\d.]+" y="\d+" width="([\d.]+)"', svg).group(1))
    assert width >= 1.0


def test_a_huge_margin_is_clipped_inside_the_drawing():
    """Bars are clipped at the axis rather than drawn outside the viewBox."""
    svg = subject_chart({"z": {"margin_over_baseline": -5.0, "has_skill": False}})
    x = float(re.search(r'<rect x="([\d.]+)"', svg).group(1))
    assert x >= 0.0


# --- rendering is total ----------------------------------------------------


def test_render_produces_a_complete_page(page: str):
    assert page.startswith("<title>")
    assert "</style>" in page
    assert page.rstrip().endswith("</div>")
    # No unresolved format placeholders left behind by an f-string edit.
    assert "{" not in page.split("</style>")[1]


def test_render_is_deterministic():
    """Same inputs, same bytes -- so a rebuild produces no spurious diff."""
    from tools.status_page import _load

    args = dict(
        harvest=_load("harvest-report.json"),
        four_class=_load("real-data-evaluation.json"),
        ablation=_load("feature-ablation.json"),
        n_tests=123,
    )
    assert render(**args) == render(**args)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
