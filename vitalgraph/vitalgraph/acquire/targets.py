"""The curated harvest list: which public repositories are worth reading.

Clone-based acquisition cannot discover repositories, so this file is the
discovery step, done by hand. Each entry says *why* it is here, because a
target list without stated intent decays into a list nobody dares prune.

The ``expected_spdx`` field is advisory only. It is never used to decide
policy -- that comes from :func:`vitalgraph.ingest.pipeline.detect_repo_license`
reading the checkout's own LICENSE file. Its purpose is the opposite: when
detection disagrees with what was expected, the harvest report says so, which
catches a relicensed upstream or a wrong guess instead of letting it pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class Category(str, Enum):
    """What a repository teaches us."""

    BLE_STACK = "ble_stack"
    """Generic Bluetooth Low Energy transport: how to connect, discover
    services, and read characteristics on each platform."""

    WEARABLE_PROTOCOL = "wearable_protocol"
    """Vendor-specific device protocols, usually established by
    reverse-engineering. The most direct answer to how a competitor's device
    actually frames its data."""

    BIOSIGNAL = "biosignal"
    """Physiological signal processing: HRV, PPG, ECG, respiration. What to do
    with the samples once they arrive."""

    HEALTH_INTEROP = "health_interop"
    """Clinical data interchange -- FHIR, LOINC, waveform formats."""


@dataclass(frozen=True, slots=True)
class Target:
    """One repository to harvest, and the reason it earns a place."""

    name: str
    url: str
    category: Category
    rationale: str
    """Why this repository is worth reading. Required -- an entry nobody can
    justify is an entry that should be removed."""

    expected_spdx: str | None = None
    """Advisory. Detection governs; a mismatch is reported, not obeyed."""

    heavy: bool = False
    """Large enough that a default run should skip it."""


#: The list. Ordered by category, then by how directly each answers the
#: question "how does a device like the one we are replacing actually work".
TARGETS: List[Target] = [
    # -- BLE transport -------------------------------------------------
    Target(
        name="bleak",
        url="https://github.com/hbldh/bleak.git",
        category=Category.BLE_STACK,
        rationale=(
            "The reference cross-platform BLE client in Python. Shows how "
            "characteristic reads and notifications are abstracted over four "
            "very different OS stacks, which is the shape our own device "
            "layer has to take."
        ),
        expected_spdx="MIT",
    ),
    Target(
        name="Android-BLE-Library",
        url="https://github.com/NordicSemiconductor/Android-BLE-Library.git",
        category=Category.BLE_STACK,
        rationale=(
            "Nordic's production Android BLE stack. The connection-state and "
            "request-queue handling here is the accumulated answer to why "
            "naive BLE code fails on real phones."
        ),
        expected_spdx="BSD-3-Clause",
    ),
    Target(
        name="IOS-CoreBluetooth-Mock",
        url="https://github.com/NordicSemiconductor/IOS-CoreBluetooth-Mock.git",
        category=Category.BLE_STACK,
        rationale=(
            "Models CoreBluetooth's peripheral behaviour precisely enough to "
            "test against. Directly relevant to our own simulator: it is the "
            "same problem, solved for iOS."
        ),
        expected_spdx="BSD-3-Clause",
    ),
    Target(
        name="Adafruit_CircuitPython_BLE",
        url="https://github.com/adafruit/Adafruit_CircuitPython_BLE.git",
        category=Category.BLE_STACK,
        rationale=(
            "Readable implementations of standard GATT services with the "
            "packing and unpacking written out explicitly, which makes it a "
            "good cross-check on our own 0x2A37 decoder."
        ),
        expected_spdx="MIT",
    ),
    # -- Vendor protocols, reverse-engineered ---------------------------
    Target(
        name="Gadgetbridge",
        url="https://github.com/Freeyourgadget/Gadgetbridge.git",
        category=Category.WEARABLE_PROTOCOL,
        rationale=(
            "The largest body of reverse-engineered wearable protocols in "
            "public: dozens of vendors' proprietary GATT services decoded "
            "without their cooperation. This is the single most relevant "
            "repository to replacing a closed device. AGPL, so the gate will "
            "hold its source back and admit only the protocol facts -- which "
            "is exactly the material we want from it."
        ),
        expected_spdx="AGPL-3.0-only",
        heavy=True,
    ),
    Target(
        name="openhaystack",
        url="https://github.com/seemoo-lab/openhaystack.git",
        category=Category.WEARABLE_PROTOCOL,
        rationale=(
            "Reverse-engineered Apple BLE advertising. Useful as method: how "
            "an undocumented advertising payload is decoded and documented."
        ),
        expected_spdx="AGPL-3.0-only",
    ),
    # -- Biosignal processing ------------------------------------------
    Target(
        name="NeuroKit",
        url="https://github.com/neuropsychology/NeuroKit.git",
        category=Category.BIOSIGNAL,
        rationale=(
            "Broad, well-cited physiological signal toolkit -- HRV, PPG, ECG, "
            "respiration. The HRV index definitions here are a check on ours, "
            "and the PPG pipeline is what wrist-based sensing actually needs."
        ),
        expected_spdx="MIT",
    ),
    Target(
        name="heartrate_analysis_python",
        url="https://github.com/paulvangentcom/heartrate_analysis_python.git",
        category=Category.BIOSIGNAL,
        rationale=(
            "HeartPy: PPG-first HRV analysis, built for noisy consumer-grade "
            "sensors rather than clinical ECG. That noise model is our case."
        ),
        expected_spdx="MIT",
    ),
    Target(
        name="wfdb-python",
        url="https://github.com/MIT-LCP/wfdb-python.git",
        category=Category.BIOSIGNAL,
        rationale=(
            "Reader for the PhysioNet waveform format. The route to real "
            "labelled physiological data, which is what our synthetic-trained "
            "models have to be replaced with before they mean anything."
        ),
        expected_spdx="MIT",
    ),
    Target(
        name="hrv-analysis",
        url="https://github.com/Aura-healthcare/hrv-analysis.git",
        category=Category.BIOSIGNAL,
        rationale=(
            "Focused RR-interval library with explicit artifact-removal "
            "strategies -- a direct comparison for our Malik-rule correction."
        ),
        expected_spdx="GPL-3.0-only",
    ),
    # -- Clinical interoperability -------------------------------------
    Target(
        name="client-py",
        url="https://github.com/smart-on-fhir/client-py.git",
        category=Category.HEALTH_INTEROP,
        rationale=(
            "SMART on FHIR Python client with the generated resource models. "
            "The concrete shape of an Observation, which is where our "
            "biometrics have to land to be interoperable."
        ),
        expected_spdx="Apache-2.0",
    ),
]


def by_name(name: str) -> Target:
    for target in TARGETS:
        if target.name == name:
            return target
    raise KeyError(f"no harvest target named {name!r}")


def by_category(category: Category) -> List[Target]:
    return [t for t in TARGETS if t.category is category]


def default_selection(include_heavy: bool = False) -> List[Target]:
    """Targets a plain harvest run should fetch."""
    return [t for t in TARGETS if include_heavy or not t.heavy]


def catalogue_summary() -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for target in TARGETS:
        counts[target.category.value] = counts.get(target.category.value, 0) + 1
    return counts
