"""Bluetooth Low Energy ingestion: decoders, UUIDs, and an optional live client.

The decoders in :mod:`.codecs` are pure functions over bytes and have no
Bluetooth dependency. :class:`~.client.BLECollector` needs ``bleak`` and is
imported lazily so that everything else keeps working without a radio.
"""

from . import codecs, uuids
from .codecs import Decoded, DecodeError, RevolutionTracker, decode

__all__ = [
    "codecs",
    "uuids",
    "Decoded",
    "DecodeError",
    "RevolutionTracker",
    "decode",
]
