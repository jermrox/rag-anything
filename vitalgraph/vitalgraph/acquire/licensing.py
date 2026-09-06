"""License identification: SPDX id -> :class:`LicenseClass`, and detection of
a license from raw text.

Detection is intentionally conservative. Every path that cannot positively
identify a license returns ``UNKNOWN``, which the gate in ``acquire/base.py``
treats as strictly as proprietary. Guessing generously here would be the one
mistake that quietly defeats the whole gate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from .base import LicenseClass

#: SPDX identifiers we classify explicitly. Anything absent is UNKNOWN.
SPDX_CLASSES: Dict[str, LicenseClass] = {
    # Permissive
    "MIT": LicenseClass.PERMISSIVE,
    "MIT-0": LicenseClass.PERMISSIVE,
    "BSD-2-Clause": LicenseClass.PERMISSIVE,
    "BSD-3-Clause": LicenseClass.PERMISSIVE,
    "Apache-2.0": LicenseClass.PERMISSIVE,
    "ISC": LicenseClass.PERMISSIVE,
    "Zlib": LicenseClass.PERMISSIVE,
    "Unlicense": LicenseClass.PERMISSIVE,
    "CC0-1.0": LicenseClass.PERMISSIVE,
    "BSL-1.0": LicenseClass.PERMISSIVE,
    "PSF-2.0": LicenseClass.PERMISSIVE,
    # Copyleft (weak and strong alike: neither may be reproduced verbatim
    # into a proprietary corpus, so the distinction does not change policy)
    "GPL-2.0-only": LicenseClass.COPYLEFT,
    "GPL-2.0-or-later": LicenseClass.COPYLEFT,
    "GPL-3.0-only": LicenseClass.COPYLEFT,
    "GPL-3.0-or-later": LicenseClass.COPYLEFT,
    "LGPL-2.1-only": LicenseClass.COPYLEFT,
    "LGPL-2.1-or-later": LicenseClass.COPYLEFT,
    "LGPL-3.0-only": LicenseClass.COPYLEFT,
    "LGPL-3.0-or-later": LicenseClass.COPYLEFT,
    "AGPL-3.0-only": LicenseClass.COPYLEFT,
    "AGPL-3.0-or-later": LicenseClass.COPYLEFT,
    "MPL-2.0": LicenseClass.COPYLEFT,
    "EPL-2.0": LicenseClass.COPYLEFT,
    "CDDL-1.0": LicenseClass.COPYLEFT,
    "CC-BY-SA-4.0": LicenseClass.COPYLEFT,
    # Explicitly non-free
    "LicenseRef-Proprietary": LicenseClass.PROPRIETARY,
    "NONE": LicenseClass.PROPRIETARY,
}

#: Phrases distinctive enough to identify a license from its text. Ordered:
#: the first match wins, so more specific licenses must precede the families
#: they resemble (AGPL before GPL, LGPL before GPL).
_TEXT_SIGNATURES: List[tuple[str, str]] = [
    ("GNU AFFERO GENERAL PUBLIC LICENSE", "AGPL-3.0-only"),
    ("GNU LESSER GENERAL PUBLIC LICENSE", "LGPL-3.0-only"),
    ("GNU GENERAL PUBLIC LICENSE", "GPL-3.0-only"),
    ("MOZILLA PUBLIC LICENSE", "MPL-2.0"),
    ("ECLIPSE PUBLIC LICENSE", "EPL-2.0"),
    ("APACHE LICENSE", "Apache-2.0"),
    ("PERMISSION IS HEREBY GRANTED, FREE OF CHARGE", "MIT"),
    ("REDISTRIBUTION AND USE IN SOURCE AND BINARY FORMS", "BSD-3-Clause"),
    ("PERMISSION TO USE, COPY, MODIFY, AND/OR DISTRIBUTE", "ISC"),
    (
        "THIS IS FREE AND UNENCUMBERED SOFTWARE RELEASED INTO THE PUBLIC DOMAIN",
        "Unlicense",
    ),
    ("BOOST SOFTWARE LICENSE", "BSL-1.0"),
]

#: Phrases that positively indicate a license forbidding redistribution.
_PROPRIETARY_SIGNATURES = (
    "ALL RIGHTS RESERVED",
    "PROPRIETARY AND CONFIDENTIAL",
    "NO PART OF THIS SOFTWARE MAY BE REPRODUCED",
    "UNAUTHORIZED COPYING",
)

#: `SPDX-License-Identifier: MIT` headers, the most reliable signal there is.
_SPDX_HEADER = re.compile(
    r"SPDX-License-Identifier:\s*([A-Za-z0-9.\-+]+)", re.IGNORECASE
)

#: GPL version distinctions, applied after the family is identified.
_VERSION_2 = re.compile(r"VERSION\s+2(?:\.1)?", re.IGNORECASE)

#: How much of a license document counts as its title block.
#:
#: A license names itself at the top and then discusses *other* licenses
#: further down. GPL-3.0 section 13 is the case that forced this: it contains
#: the words "GNU Affero General Public License", so a whole-document search
#: identifies every GPL-3.0 project as AGPL-3.0. Restricting title matching to
#: the opening of the file is what distinguishes "this is the licence" from
#: "this licence mentions another one".
#:
#: 2000 characters clears the longest title block seen in practice (Apache-2.0
#: reaches its distinctive phrasing at roughly 500) while stopping well short
#: of GPL-3.0's section 13 at roughly 30000.
TITLE_WINDOW_CHARS = 2000


@dataclass(frozen=True, slots=True)
class LicenseFinding:
    """The result of identifying a license."""

    spdx_id: str | None
    license_class: LicenseClass
    confidence: float
    """0.0-1.0. An explicit SPDX header is 1.0; text matching is lower, because
    a README quoting a license is not the same as being under it."""
    evidence: str
    """Why this conclusion was reached -- shown in audit output."""


def classify_spdx(spdx_id: str | None) -> LicenseClass:
    """Map an SPDX identifier to a license class.

    Unrecognised identifiers return UNKNOWN rather than a guess.
    """
    if not spdx_id:
        return LicenseClass.UNKNOWN
    normalized = spdx_id.strip()
    if normalized in SPDX_CLASSES:
        return SPDX_CLASSES[normalized]
    # Case-insensitive retry before giving up.
    lowered = normalized.lower()
    for known, cls in SPDX_CLASSES.items():
        if known.lower() == lowered:
            return cls
    return LicenseClass.UNKNOWN


def detect_from_spdx_header(text: str) -> LicenseFinding | None:
    """Find an ``SPDX-License-Identifier`` header. The strongest signal."""
    match = _SPDX_HEADER.search(text)
    if not match:
        return None
    spdx_id = match.group(1)
    return LicenseFinding(
        spdx_id=spdx_id,
        license_class=classify_spdx(spdx_id),
        confidence=1.0,
        evidence=f"SPDX-License-Identifier header: {spdx_id}",
    )


def detect_license(text: str) -> LicenseFinding:
    """Identify the license governing ``text``.

    Tries an explicit SPDX header first, then distinctive license phrasing,
    then explicit proprietary notices. Returns UNKNOWN when nothing matches --
    which the gate treats as restrictively as proprietary.
    """
    if not text or not text.strip():
        return LicenseFinding(None, LicenseClass.UNKNOWN, 0.0, "empty content")

    header = detect_from_spdx_header(text)
    if header is not None:
        return header

    upper = text.upper()
    title_block = upper[:TITLE_WINDOW_CHARS]

    # Two passes over the same signatures. The first looks only at the title
    # block, where a document names itself; the second falls back to the whole
    # text for files that bury or omit a conventional title. Confidence is
    # lower on the fallback because a phrase deep in a document is as likely to
    # be a cross-reference as a self-description.
    for haystack, confidence, where in (
        (title_block, 0.8, "title block"),
        (upper, 0.6, "document body"),
    ):
        for phrase, spdx_id in _TEXT_SIGNATURES:
            if phrase not in haystack:
                continue
            # GPL-family texts name their version in the body.
            if spdx_id.startswith(("GPL", "LGPL", "AGPL")) and _VERSION_2.search(upper):
                spdx_id = spdx_id.replace("3.0", "2.0").replace("LGPL-2.0", "LGPL-2.1")
            return LicenseFinding(
                spdx_id=spdx_id,
                license_class=classify_spdx(spdx_id),
                confidence=confidence,
                evidence=f"matched license text in {where}: {phrase[:48].title()}",
            )

    for phrase in _PROPRIETARY_SIGNATURES:
        if phrase in upper:
            return LicenseFinding(
                spdx_id="LicenseRef-Proprietary",
                license_class=LicenseClass.PROPRIETARY,
                confidence=0.7,
                evidence=f"proprietary notice: {phrase[:48].title()}",
            )

    return LicenseFinding(
        None,
        LicenseClass.UNKNOWN,
        0.0,
        "no recognised license text; treated as restrictively as proprietary",
    )


#: Filenames conventionally holding a project's license.
LICENSE_FILENAMES = (
    "LICENSE",
    "LICENSE.txt",
    "LICENSE.md",
    "LICENCE",
    "LICENCE.txt",
    "COPYING",
    "COPYING.txt",
    "LICENSE-MIT",
    "LICENSE-APACHE",
)
