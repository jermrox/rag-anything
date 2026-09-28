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

    MEDICAL_DEVICE = "medical_device"
    """Code that talks to a regulated or clinical-grade body-worn device.

    Separate from WEARABLE_PROTOCOL because the engineering is different in
    kind, not degree. A consumer band that drops a packet shows a wrong step
    count; a CGM that drops one can hide a hypo. So this code carries things
    the consumer stacks do not: calibration state machines, backfill of
    missed readings, sensor-session lifecycle, staleness rules that refuse to
    display an old value, and explicit handling of a sensor that has failed
    rather than merely gone quiet.

    That last set is the reason this category exists for us. Our own signal
    model already refuses to invent values, and these are the codebases where
    that discipline has been tested against a device people's health depends
    on. They are also the clearest published examples of vendor BLE protocols
    recovered by reverse-engineering, which is the same problem as reading an
    undocumented band."""

    SLEEP_STAGING = "sleep_staging"
    """Turning signals into sleep stages, and evaluating the result honestly.

    A category of its own rather than a corner of BIOSIGNAL. Sleep is the
    factor a wrist device covers best and the one where our own stager is
    weakest -- it is inferred, not measured -- so what the published work does
    about validation, class imbalance and subject-level splitting is the part
    worth reading. It was also the first gap the corpus search exposed: a
    query for "sleep staging" returned a thread-sleep helper and a smart-bulb
    example, because nothing in the target list was about sleep at all."""


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
    # -- Sleep staging --------------------------------------------------
    Target(
        name="sleep_classifiers",
        url="https://github.com/ojwalch/sleep_classifiers.git",
        category=Category.SLEEP_STAGING,
        rationale=(
            "Walch et al.'s wrist sleep staging from motion and heart rate "
            "alone -- exactly our sensor set, on exactly our wrist, scored "
            "against polysomnography. The most directly comparable published "
            "work to what our own stager claims, and the reference our "
            "synthetic-trained model has to be replaced by. Note that it ships "
            "no LICENSE file -- its README states MIT in prose -- so the gate "
            "resolves it to UNKNOWN and admits only its methods and documented "
            "behaviour. That is the correct outcome and not a limitation worth "
            "working around: what we want from it is the approach, not its "
            "source text."
        ),
        expected_spdx=None,
    ),
    Target(
        name="yasa",
        url="https://github.com/raphaelvallat/yasa.git",
        category=Category.SLEEP_STAGING,
        rationale=(
            "Widely used automated staging with published validation and "
            "explicit confidence output. Its feature set and its willingness "
            "to report per-epoch uncertainty are both things our stager "
            "currently lacks."
        ),
        expected_spdx="BSD-3-Clause",
    ),
    Target(
        name="deepsleepnet",
        url="https://github.com/akaraspt/deepsleepnet.git",
        category=Category.SLEEP_STAGING,
        rationale=(
            "A well-cited staging network trained on Sleep-EDF, useful mainly "
            "for how it handles class imbalance -- N1 is rare and easy to "
            "score well by never predicting, which is the failure our own "
            "100%-accuracy result was an instance of."
        ),
        expected_spdx="Apache-2.0",
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
    # -- medical and clinical-grade devices -----------------------------
    Target(
        name="polar-ble-sdk",
        url="https://github.com/polarofficial/polar-ble-sdk.git",
        category=Category.MEDICAL_DEVICE,
        rationale=(
            "The vendor's own SDK for the H10, the chest strap research "
            "studies use as ground truth for RR intervals. Documents how a "
            "manufacturer exposes raw ECG, accelerometry and per-beat "
            "intervals through custom characteristics alongside the standard "
            "0x2A37 -- the difference between a band that gives you a number "
            "and one that gives you the signal."
        ),
        expected_spdx=None,
    ),
    Target(
        name="bleakheart",
        url="https://github.com/fsmeraldi/bleakheart.git",
        category=Category.MEDICAL_DEVICE,
        rationale=(
            "Reads Polar ECG and accelerometer streams from Python without "
            "the vendor SDK. The independent reimplementation is worth more "
            "than the SDK for our purposes: it shows which parts of the "
            "protocol are actually necessary, and it is built on bleak, "
            "which we already read."
        ),
        expected_spdx="MIT",
    ),
    Target(
        name="movesense-ble-ecg-firmware",
        url="https://github.com/JonasPrimbs/movesense-ble-ecg-firmware.git",
        category=Category.MEDICAL_DEVICE,
        rationale=(
            "Firmware, not a client. The other side of the connection: how a "
            "medical-grade sensor defines a custom GATT service, packs ECG "
            "voltages, and manages notification rate against battery. The "
            "only target that answers what a device could expose if we were "
            "designing one."
        ),
    ),
    Target(
        name="xdrip",
        url="https://github.com/NightscoutFoundation/xDrip.git",
        category=Category.MEDICAL_DEVICE,
        rationale=(
            "A decade of reverse-engineered CGM transmitter protocols in one "
            "Android codebase -- Dexcom, Libre, and more -- with the "
            "calibration, backfill and sensor-session logic around them. The "
            "single richest public example of talking to an undocumented "
            "medical device over BLE and refusing to display a reading it "
            "cannot stand behind."
        ),
        expected_spdx="GPL-3.0",
        heavy=True,
    ),
    Target(
        name="cgm-remote-monitor",
        url="https://github.com/nightscout/cgm-remote-monitor.git",
        category=Category.MEDICAL_DEVICE,
        rationale=(
            "What happens to device data after it leaves the device: "
            "staleness rules, alarm escalation, and a display that says the "
            "data is old rather than showing the last value as if current. "
            "That is the same problem as our evidence tiers, solved in "
            "production for people who act on it."
        ),
        expected_spdx="AGPL-3.0",
    ),
    Target(
        name="oref0",
        url="https://github.com/openaps/oref0.git",
        category=Category.MEDICAL_DEVICE,
        rationale=(
            "A closed-loop dosing algorithm that has to be safe when its "
            "inputs are missing, stale or wrong. Read for its refusal "
            "logic rather than its dosing: the conditions under which it "
            "declines to act are the most rigorously argued 'we do not know "
            "enough' code in open-source health software."
        ),
        expected_spdx="MIT",
    ),
    Target(
        name="EmotiBit_FeatherWing",
        url="https://github.com/EmotiBit/EmotiBit_FeatherWing.git",
        category=Category.MEDICAL_DEVICE,
        rationale=(
            "Open hardware and firmware streaming multi-wavelength PPG, "
            "nine-axis motion, skin temperature and EDA. Multi-wavelength "
            "PPG is what separates a research sensor from a consumer band, "
            "and this is the readable implementation of it."
        ),
        expected_spdx="See-License-File",
    ),
    Target(
        name="open-wearables",
        url="https://github.com/the-momentum/open-wearables.git",
        category=Category.MEDICAL_DEVICE,
        rationale=(
            "Normalises several vendors' health APIs behind one schema. The "
            "value is the mapping itself: where two manufacturers disagree "
            "about what a field means, this is where that disagreement has "
            "already been written down."
        ),
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
