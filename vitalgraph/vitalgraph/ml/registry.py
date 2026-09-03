"""Model persistence, versioning and provenance.

Every persisted model carries a :class:`ModelCard`. The field that matters most
is :attr:`ModelCard.training_data`: a model trained on ``ble/simulator.py``
output has learned the simulator, not human physiology, and will score
beautifully on held-out synthetic nights while meaning very little about real
wrists. Recording that on the model itself -- rather than in a README nobody
reads at inference time -- is what stops a scaffold being mistaken for a
validated instrument.

Models are stored as pickles because that is what scikit-learn estimators
support. Pickle executes arbitrary code on load, so :meth:`ModelRegistry.load`
reads only from the registry directory the process itself configured, and the
format is unsuitable for accepting models from untrusted sources.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .features import FEATURE_NAMES

#: Training-data provenance values.
SYNTHETIC = "synthetic"
"""Trained on simulator output. Proves the pipeline; not clinically meaningful."""

REAL = "real"
"""Trained on measured human data."""

MIXED = "mixed"


@dataclass
class ModelCard:
    """What a model is, how it was made, and what it may be trusted for."""

    name: str
    algorithm: str
    training_data: str
    """One of SYNTHETIC / REAL / MIXED. See the module docstring."""
    version: str = "1"
    trained_at: str = ""
    seed: int | None = None
    feature_names: Tuple[str, ...] = FEATURE_NAMES
    n_training_samples: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.training_data not in (SYNTHETIC, REAL, MIXED):
            raise ValueError(
                f"training_data must be one of {SYNTHETIC!r}, {REAL!r}, {MIXED!r}"
            )
        if not self.trained_at:
            self.trained_at = datetime.now(timezone.utc).isoformat()

    @property
    def is_synthetic(self) -> bool:
        return self.training_data == SYNTHETIC

    @property
    def qualified_name(self) -> str:
        return f"{self.name}-v{self.version}"

    def caveat(self) -> str:
        """One-line honesty statement, surfaced with every prediction."""
        if self.is_synthetic:
            return (
                "Trained on simulated data: this model has learned the simulator's "
                "assumptions, not human physiology. Use it to validate the pipeline, "
                "not to interpret a real person's health."
            )
        return (
            f"Trained on {self.training_data} data ({self.n_training_samples} samples)."
        )

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["feature_names"] = list(self.feature_names)
        d["caveat"] = self.caveat()
        return d


class ModelNotFound(KeyError):
    """Raised when a requested model is not in the registry."""


class FeatureMismatch(ValueError):
    """Raised when a model's features do not match the current vocabulary.

    Loading a model trained against a different feature order and using it
    anyway produces confident nonsense, so it is refused outright.
    """


class ModelRegistry:
    """Directory-backed store of models and their cards."""

    def __init__(self, root: str | Path = "./vitalgraph_models") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _paths(self, name: str) -> Tuple[Path, Path]:
        return self.root / f"{name}.pkl", self.root / f"{name}.json"

    def save(self, model: Any, card: ModelCard) -> Path:
        """Persist a model with its card."""
        model_path, card_path = self._paths(card.name)
        with model_path.open("wb") as fh:
            pickle.dump(model, fh, protocol=pickle.HIGHEST_PROTOCOL)
        card_path.write_text(json.dumps(card.as_dict(), indent=2))
        return model_path

    def load(self, name: str, strict_features: bool = True) -> Tuple[Any, ModelCard]:
        """Load a model and its card.

        Args:
            name: registered model name.
            strict_features: refuse a model whose recorded feature vocabulary
                differs from the current one. Disabling this is only sensible
                for inspection, never for inference.
        """
        model_path, card_path = self._paths(name)
        if not model_path.exists() or not card_path.exists():
            raise ModelNotFound(f"no model named {name!r} in {self.root}")

        raw = json.loads(card_path.read_text())
        raw.pop("caveat", None)
        raw["feature_names"] = tuple(raw.get("feature_names", ()))
        card = ModelCard(**raw)

        if strict_features and tuple(card.feature_names) != FEATURE_NAMES:
            raise FeatureMismatch(
                f"model {name!r} was trained on a different feature vocabulary "
                f"({len(card.feature_names)} features); retrain it before use"
            )

        with model_path.open("rb") as fh:
            model = pickle.load(fh)
        return model, card

    def card(self, name: str) -> ModelCard:
        _, card_path = self._paths(name)
        if not card_path.exists():
            raise ModelNotFound(f"no model named {name!r} in {self.root}")
        raw = json.loads(card_path.read_text())
        raw.pop("caveat", None)
        raw["feature_names"] = tuple(raw.get("feature_names", ()))
        return ModelCard(**raw)

    def list_models(self) -> List[Dict[str, Any]]:
        out = []
        for card_path in sorted(self.root.glob("*.json")):
            raw = json.loads(card_path.read_text())
            out.append(
                {
                    "name": raw.get("name"),
                    "version": raw.get("version"),
                    "algorithm": raw.get("algorithm"),
                    "training_data": raw.get("training_data"),
                    "trained_at": raw.get("trained_at"),
                    "metrics": raw.get("metrics", {}),
                    "caveat": raw.get("caveat", ""),
                }
            )
        return out

    def delete(self, name: str) -> bool:
        model_path, card_path = self._paths(name)
        existed = model_path.exists() or card_path.exists()
        model_path.unlink(missing_ok=True)
        card_path.unlink(missing_ok=True)
        return existed


def require_sklearn() -> Any:
    """Import scikit-learn, or explain how to get it.

    The analytics core is standard-library only by design, so the ML extra is
    genuinely optional and its absence must produce a clear instruction rather
    than an ImportError from three frames down.
    """
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "This feature needs the ML extra. Install it with:\n"
            '    pip install -e "vitalgraph[ml]"'
        ) from exc
    return sklearn
