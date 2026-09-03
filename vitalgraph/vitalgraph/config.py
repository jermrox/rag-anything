"""Environment-backed configuration.

Mirrors the style of ``raganything/config.py`` -- dataclass fields with env
defaults -- so anyone who knows the host framework recognises the pattern.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


@dataclass
class VitalGraphConfig:
    """Runtime configuration for the VitalGraph subproject."""

    data_dir: str = field(
        default_factory=lambda: _env("VITALGRAPH_DATA_DIR", "./vitalgraph_data")
    )
    """Directory holding the biometric SQLite database."""

    db_path: str = field(
        default_factory=lambda: _env("VITALGRAPH_DB", "./vitalgraph_data/biometrics.db")
    )

    rag_working_dir: str = field(
        default_factory=lambda: _env("VITALGRAPH_RAG_DIR", "./vg_rag_storage")
    )

    user: str = field(default_factory=lambda: _env("VITALGRAPH_USER", "default"))

    baseline_nights: int = field(
        default_factory=lambda: _env_int("VITALGRAPH_BASELINE_NIGHTS", 14)
    )
    """Rolling window used for the personal HRV baseline."""

    api_host: str = field(default_factory=lambda: _env("VITALGRAPH_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _env_int("VITALGRAPH_PORT", 8770))
