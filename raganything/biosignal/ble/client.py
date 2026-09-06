"""Live BLE collection: discover what a device actually offers, then take it all.

Most fitness apps connect to a device looking for one characteristic they
already decided they wanted. This collector inverts that: it enumerates every
characteristic the peripheral exposes, subscribes to everything it can decode,
and records everything it cannot as opaque payloads for later analysis. Devices
routinely publish more than their own apps display, and you cannot find that
out by asking only for what you expected.

``bleak`` is an optional dependency -- everything in :mod:`.codecs` works
without a radio, which is what makes this subsystem testable in CI.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..schema import Modality, Provenance, Session, SourceKind, Stream
from . import uuids
from .codecs import DecodeError, Decoded, decode

logger = logging.getLogger(__name__)

__all__ = ["BLECollector", "StreamRecorder", "UnknownPayload", "uuid16_of"]


def uuid16_of(full_uuid: str) -> Optional[int]:
    """Extract the 16-bit assigned number from a full UUID, if it is one."""
    text = str(full_uuid).lower()
    if not text.endswith(uuids.BASE_UUID_SUFFIX):
        return None
    try:
        return int(text[4:8], 16)
    except ValueError:
        return None


@dataclass
class UnknownPayload:
    """A notification we could not decode, kept rather than dropped."""

    t: float
    characteristic: str
    data: bytes


@dataclass
class StreamRecorder:
    """Accumulates decoded readings into per-source, per-modality streams."""

    source_id: str
    device: str = "unknown"
    streams: Dict[Tuple[Modality, int], Stream] = field(default_factory=dict)
    unknown: List[UnknownPayload] = field(default_factory=list)
    raw_log: List[Dict[str, Any]] = field(default_factory=list)
    keep_raw: bool = True

    def ingest(self, decoded: Decoded) -> None:
        for modality, sample in decoded.readings:
            key = (modality, decoded.characteristic)
            stream = self.streams.get(key)
            if stream is None:
                stream = Stream(
                    modality=modality,
                    provenance=Provenance(
                        source_id=f"{self.source_id}/0x{decoded.characteristic:04X}",
                        kind=SourceKind.BLE,
                        device=self.device,
                        transport=(
                            f"GATT notify 0x{decoded.characteristic:04X} "
                            f"({uuids.name_for(decoded.characteristic)})"
                        ),
                        latency_s=0.05,
                        documented=True,
                    ),
                )
                self.streams[key] = stream
            stream.add(sample)
        if self.keep_raw and decoded.raw:
            self.raw_log.append({"t": decoded.t, **decoded.raw})

    def note_unknown(self, t: float, characteristic: str, data: bytes) -> None:
        self.unknown.append(
            UnknownPayload(t=t, characteristic=characteristic, data=bytes(data))
        )

    def to_session(
        self,
        session_id: str,
        start: Optional[float] = None,
        end: Optional[float] = None,
        labels: Optional[Dict[str, Any]] = None,
    ) -> Session:
        streams = [s.sorted().flag_implausible() for s in self.streams.values()]
        starts = [s.start for s in streams if s.start is not None]
        ends = [s.end for s in streams if s.end is not None]
        session = Session(
            session_id=session_id,
            start=start if start is not None else (min(starts) if starts else 0.0),
            end=end if end is not None else (max(ends) if ends else 0.0),
            streams=streams,
            labels=dict(labels or {}),
        )
        if self.unknown:
            session.labels.setdefault(
                "undecoded_notifications",
                {
                    "count": len(self.unknown),
                    "characteristics": sorted({u.characteristic for u in self.unknown}),
                },
            )
        return session


class BLECollector:
    """Connect to one peripheral and record everything it will tell us.

    Args:
        address: BLE address or platform device identifier.
        device_name: Human label carried into provenance.
        on_decoded: Optional callback fired for every decoded notification --
            use it for live UI without waiting for the session to end.
    """

    def __init__(
        self,
        address: str,
        device_name: str = "unknown",
        on_decoded: Optional[Callable[[Decoded], None]] = None,
        keep_raw: bool = True,
    ) -> None:
        self.address = address
        self.device_name = device_name
        self.on_decoded = on_decoded
        self.recorder = StreamRecorder(
            source_id=address, device=device_name, keep_raw=keep_raw
        )
        self._client: Any = None

    # -- discovery -------------------------------------------------------

    @staticmethod
    async def scan(timeout: float = 8.0) -> List[Dict[str, Any]]:
        """Discover nearby peripherals, annotated with what we could read.

        Returns a list of dicts with ``address``, ``name`` and the advertised
        service UUIDs, with ``supported`` naming the ones this package decodes.
        """
        BleakScanner = _import_bleak().BleakScanner
        found = await BleakScanner.discover(timeout=timeout, return_adv=True)
        results: List[Dict[str, Any]] = []
        for device, adv in found.values():
            service_ids = [uuid16_of(u) for u in (adv.service_uuids or [])]
            supported = [uuids.name_for(s) for s in service_ids if s in _SERVICE_HINTS]
            results.append(
                {
                    "address": device.address,
                    "name": adv.local_name or device.name or "unknown",
                    "rssi": adv.rssi,
                    "service_uuids": list(adv.service_uuids or []),
                    "supported_services": supported,
                }
            )
        return results

    # -- collection ------------------------------------------------------

    async def collect(
        self,
        duration_s: float,
        characteristics: Optional[Sequence[int]] = None,
        connect_timeout: float = 15.0,
    ) -> Session:
        """Subscribe and record for ``duration_s`` seconds, then disconnect.

        Args:
            duration_s: How long to record.
            characteristics: 16-bit assigned numbers to subscribe to. Defaults
                to every notifiable characteristic the peripheral exposes --
                including ones with no decoder, whose payloads are retained raw.
            connect_timeout: Passed to the underlying client.
        """
        BleakClient = _import_bleak().BleakClient
        wanted = set(characteristics) if characteristics else None
        subscribed: List[str] = []

        async with BleakClient(self.address, timeout=connect_timeout) as client:
            self._client = client
            for service in client.services:
                for char in service.characteristics:
                    if (
                        "notify" not in char.properties
                        and "indicate" not in char.properties
                    ):
                        continue
                    short = uuid16_of(char.uuid)
                    if wanted is not None and short not in wanted:
                        continue
                    try:
                        await client.start_notify(char.uuid, self._make_handler(short))
                        subscribed.append(char.uuid)
                    except Exception as exc:  # noqa: BLE001 - device-specific failures
                        logger.warning(
                            "could not subscribe to %s (%s): %s",
                            char.uuid,
                            uuids.name_for(short) if short else "vendor-specific",
                            exc,
                        )

            if not subscribed:
                raise RuntimeError(
                    f"{self.address} exposed no subscribable characteristics matching "
                    f"{sorted(wanted) if wanted else 'any'}"
                )
            logger.info(
                "collecting from %s for %.0fs across %d characteristic(s)",
                self.device_name,
                duration_s,
                len(subscribed),
            )
            started = time.time()
            try:
                await asyncio.sleep(duration_s)
            finally:
                for uuid_str in subscribed:
                    try:
                        await client.stop_notify(uuid_str)
                    except Exception:  # noqa: BLE001 - already disconnected
                        pass

        return self.recorder.to_session(
            session_id=f"{self.device_name}-{int(started)}",
            start=started,
            end=time.time(),
        )

    def _make_handler(
        self, short_uuid: Optional[int]
    ) -> Callable[[Any, bytearray], None]:
        def handler(sender: Any, data: bytearray) -> None:
            t = time.time()
            if short_uuid is None:
                self.recorder.note_unknown(t, str(sender), data)
                return
            try:
                decoded = decode(short_uuid, bytes(data), t)
            except DecodeError as exc:
                logger.debug("undecodable payload on 0x%04X: %s", short_uuid, exc)
                self.recorder.note_unknown(t, f"0x{short_uuid:04X}", data)
                return
            self.recorder.ingest(decoded)
            if self.on_decoded is not None:
                self.on_decoded(decoded)

        return handler


_SERVICE_HINTS = {
    uuids.SVC_HEART_RATE,
    uuids.SVC_CYCLING_POWER,
    uuids.SVC_CYCLING_SPEED_CADENCE,
    uuids.SVC_RUNNING_SPEED_CADENCE,
    uuids.SVC_FITNESS_MACHINE,
    uuids.SVC_BODY_COMPOSITION,
    uuids.SVC_WEIGHT_SCALE,
    uuids.SVC_GLUCOSE,
    uuids.SVC_CONTINUOUS_GLUCOSE,
    uuids.SVC_PULSE_OXIMETER,
    uuids.SVC_BATTERY,
}


def _import_bleak():
    try:
        import bleak  # noqa: PLC0415 - optional dependency, imported on demand
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Live BLE collection requires the optional 'bleak' dependency:\n"
            "    pip install raganything[biosignal]\n"
            "Decoding, analytics and indexing work without it."
        ) from exc
    return bleak
