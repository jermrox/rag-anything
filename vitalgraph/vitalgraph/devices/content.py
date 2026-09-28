"""Rendering the device catalogue as knowledge-graph content.

Follows the same discipline as ``bridge/summarizer.py`` and
``knowledge/rag_domains.py``: deterministic rendering from the catalogue, so
the fact that Whoop's radio link is closed while its API is open is something
the RAG can be *asked about* and answer correctly -- not a footnote that only
lives in a docstring.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .catalog import DEVICES, AccessMode, DeviceProfile


def _reachability_row(device: DeviceProfile) -> str:
    mode = device.access_mode.value
    path = (
        ", ".join(device.gatt_services)
        if device.access_mode is AccessMode.OPEN_BLE
        else (device.api_docs_url or "none")
    )
    return f"| {device.name} | {device.vendor} | {mode} | {path} |"


def device_landscape_content_list() -> List[Dict[str, Any]]:
    """The full device catalogue as insertable content: one narrative
    per access mode, plus a reachability table."""
    open_ble = [d for d in DEVICES if d.access_mode is AccessMode.OPEN_BLE]
    cloud = [d for d in DEVICES if d.access_mode is AccessMode.CLOUD_API]
    closed = [d for d in DEVICES if d.access_mode is AccessMode.CLOSED]

    narrative = [
        f"Device landscape: {len(DEVICES)} named products across three "
        f"reachability classes.",
        f"Directly connectable over open Bluetooth GATT, no vendor "
        f"involved ({len(open_ble)}): " + ", ".join(d.name for d in open_ble) + ".",
        f"Reachable only through the vendor's own cloud API after user "
        f"OAuth consent ({len(cloud)}): "
        + ", ".join(d.name for d in cloud)
        + ". Their underlying radio protocols are proprietary and are not "
        "decoded directly.",
    ]
    if closed:
        narrative.append(
            f"No third-party path today, direct or via API ({len(closed)}): "
            + ", ".join(d.name for d in closed)
            + "."
        )
    narrative.append(
        "A cloud-API device's sensor is not upgraded by having an API: "
        "Whoop and Fitbit both expose wrist PPG through their APIs, so the "
        "same signal-adequacy limits that apply to any wrist-PPG sensor "
        "apply to what their APIs can honestly support."
    )

    rows = ["| Device | Vendor | Access | Path |", "| --- | --- | --- | --- |"]
    rows.extend(_reachability_row(d) for d in DEVICES)

    for device in DEVICES:
        if device.notes:
            narrative.append(f"On {device.name}: {device.notes}")

    return [
        {"type": "text", "text": "\n".join(narrative), "page_idx": 0},
        {
            "type": "table",
            "table_body": "\n".join(rows),
            "table_caption": ["Device reachability"],
            "table_footnote": [
                "OPEN_BLE devices are decoded directly by this product. "
                "CLOUD_API devices require the vendor's own authorized "
                "integration; their radio protocol is not decoded by us. "
                "CLOSED devices have no third-party path today."
            ],
            "page_idx": 0,
        },
    ]
