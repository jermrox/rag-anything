"""License identification and the gate that depends on it.

The gate is the one mechanism standing between a copyleft repository and a
proprietary product's corpus, so its failure modes are tested explicitly --
especially the direction where it fails *open*.
"""

import pytest

from vitalgraph.acquire.base import (
    LicenseClass,
    Provenance,
    UsePolicy,
    local_provenance,
    policy_for,
    summarise_licensing,
    Artifact,
)
from vitalgraph.acquire import licensing
from vitalgraph.acquire.licensing import classify_spdx, detect_license


@pytest.mark.parametrize(
    "spdx,expected",
    [
        ("MIT", LicenseClass.PERMISSIVE),
        ("Apache-2.0", LicenseClass.PERMISSIVE),
        ("BSD-3-Clause", LicenseClass.PERMISSIVE),
        ("GPL-3.0-only", LicenseClass.COPYLEFT),
        ("AGPL-3.0-only", LicenseClass.COPYLEFT),
        ("LGPL-2.1-or-later", LicenseClass.COPYLEFT),
        ("MPL-2.0", LicenseClass.COPYLEFT),
    ],
)
def test_known_spdx_ids_are_classified(spdx, expected):
    assert classify_spdx(spdx) == expected


def test_unknown_spdx_is_not_guessed():
    assert classify_spdx("Totally-Made-Up-1.0") == LicenseClass.UNKNOWN
    assert classify_spdx(None) == LicenseClass.UNKNOWN
    assert classify_spdx("") == LicenseClass.UNKNOWN


def test_spdx_header_is_the_strongest_signal():
    f = detect_license("# SPDX-License-Identifier: Apache-2.0\nprint(1)")
    assert f.spdx_id == "Apache-2.0"
    assert f.confidence == 1.0


def test_detects_licenses_from_body_text():
    assert (
        detect_license(
            "Permission is hereby granted, free of charge, to any person"
        ).license_class
        is LicenseClass.PERMISSIVE
    )
    assert (
        detect_license(
            "GNU GENERAL PUBLIC LICENSE Version 3, 29 June 2007"
        ).license_class
        is LicenseClass.COPYLEFT
    )


def test_agpl_is_not_mistaken_for_gpl():
    """AGPL text contains 'GENERAL PUBLIC LICENSE', so ordering matters."""
    f = detect_license("GNU AFFERO GENERAL PUBLIC LICENSE Version 3, 19 November 2007")
    assert f.spdx_id == "AGPL-3.0-only"


def test_gpl3_is_not_mistaken_for_agpl_because_it_mentions_agpl():
    """GPL-3.0 section 13 names the Affero GPL. That is a cross-reference.

    Regression: the first real harvest classified every GPL-3.0 project as
    AGPL-3.0, because a whole-document search found section 13's wording. A
    licence names itself at the top and discusses other licences further down,
    so title matching is confined to the opening of the file.
    """
    gpl3 = (
        "                    GNU GENERAL PUBLIC LICENSE\n"
        "                       Version 3, 29 June 2007\n\n"
        + "Preamble and terms and conditions follow.\n"
        * 400
        + "  13. Use with the GNU Affero General Public License.\n"
        "  Notwithstanding any other provision of this License, you have\n"
        "permission to link or combine any covered work with a work licensed\n"
        "under version 3 of the GNU Affero General Public License.\n"
    )
    assert len(gpl3) > licensing.TITLE_WINDOW_CHARS * 2, "fixture must be long enough"

    finding = detect_license(gpl3)
    assert finding.spdx_id == "GPL-3.0-only"
    assert finding.license_class is LicenseClass.COPYLEFT


def test_body_only_match_is_reported_with_lower_confidence():
    """A signature found outside the title block is a weaker signal, and says so."""
    buried = (
        "Copyright notice.\n" * 200 + "\nPermission is hereby granted, free of charge"
    )
    finding = detect_license(buried)
    assert finding.spdx_id == "MIT"
    assert finding.confidence < 0.8
    assert "document body" in finding.evidence


def test_lgpl_is_not_mistaken_for_gpl():
    f = detect_license("GNU LESSER GENERAL PUBLIC LICENSE Version 3, 29 June 2007")
    assert f.spdx_id.startswith("LGPL")


def test_proprietary_notice_is_detected():
    f = detect_license("Copyright Acme Inc. All rights reserved.")
    assert f.license_class is LicenseClass.PROPRIETARY


def test_unrecognised_text_is_unknown_not_permissive():
    """Failing open here would silently defeat the entire gate."""
    f = detect_license("this file contains no license information at all")
    assert f.license_class is LicenseClass.UNKNOWN
    assert f.confidence == 0.0


def test_empty_content_is_unknown():
    assert detect_license("").license_class is LicenseClass.UNKNOWN


# --- the gate itself -------------------------------------------------------


def test_only_permissive_allows_verbatim():
    assert policy_for(LicenseClass.PERMISSIVE) is UsePolicy.VERBATIM
    for cls in (LicenseClass.COPYLEFT, LicenseClass.PROPRIETARY, LicenseClass.UNKNOWN):
        assert policy_for(cls) is UsePolicy.FACTS_ONLY


def test_provenance_exposes_the_policy():
    assert local_provenance(
        "/a.py", "x", LicenseClass.PERMISSIVE, "MIT"
    ).allows_verbatim
    assert not local_provenance(
        "/a.py", "x", LicenseClass.COPYLEFT, "GPL-3.0-only"
    ).allows_verbatim
    assert not local_provenance("/a.py", "x").allows_verbatim  # defaults matter


def test_provenance_requires_timezone_aware_timestamp():
    from datetime import datetime

    with pytest.raises(ValueError):
        Provenance(
            source_url="x", retrieved_at=datetime(2026, 1, 1), content_sha256="abc"
        )


def test_content_hash_is_stable_and_content_dependent():
    a = local_provenance("/a.py", "same")
    b = local_provenance("/b.py", "same")
    c = local_provenance("/a.py", "different")
    assert a.content_sha256 == b.content_sha256
    assert a.content_sha256 != c.content_sha256


def test_attribution_names_source_and_licence():
    p = local_provenance(
        "/a.py", "x", LicenseClass.PERMISSIVE, "MIT", upstream_ref="abc123"
    )
    text = p.attribution()
    assert "MIT" in text and "abc123" in text


def test_licensing_summary_reports_policy_per_class():
    artifacts = [
        Artifact("a", "x", local_provenance("/a", "x", LicenseClass.PERMISSIVE, "MIT")),
        Artifact(
            "b", "y", local_provenance("/b", "y", LicenseClass.COPYLEFT, "GPL-3.0-only")
        ),
        Artifact(
            "c", "z", local_provenance("/c", "z", LicenseClass.COPYLEFT, "GPL-3.0-only")
        ),
    ]
    rows = {r["license_class"]: r for r in summarise_licensing(artifacts)}
    assert rows["copyleft"]["count"] == 2
    assert rows["copyleft"]["policy"] == "facts_only"
    assert rows["permissive"]["policy"] == "verbatim"
