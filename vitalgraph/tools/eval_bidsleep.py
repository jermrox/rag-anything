"""Score the stager against BIDSLEEP: wrist motion plus heart rate, 47 people.

Held out one **subject** at a time, never one night. A person contributes up
to seven nights and their physiology is the same on all of them, so splitting
by night would put the same wrist on both sides of the split and report a
number that says nothing about a stranger.

Three feature sets are run so the motion question gets a direct answer:

    heart rate only     what slpdb could see -- the honest control
    motion only         can a wrist stage sleep without a pulse at all?
    both                the whole feature set

If "both" does not beat "heart rate only", the accelerometer was not the
missing piece and the diagnosis that sent us to this dataset was wrong. That
is a result worth having either way, which is why the control is run rather
than assumed.

Usage::

    python -m tools.eval_bidsleep
    python -m tools.eval_bidsleep --no-context      # skip temporal smoothing
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from vitalgraph.biometrics.schema import SleepStage
from vitalgraph.data.bidsleep import DEFAULT_CACHE, FEATURE_NAMES
from vitalgraph.ml.context import add_temporal_context
from vitalgraph.ml.epochs import EpochSample
from vitalgraph.ml.metrics import (
    IMPLAUSIBLE_SLEEP_WAKE_ACCURACY,
    CrossValidationSummary,
    assess_skill,
)
from vitalgraph.ml.staging import collapse_to_sleep_wake

ROOT = Path(__file__).resolve().parent.parent

#: Feature subsets, by name. Indices are resolved from FEATURE_NAMES so a
#: reordering of the feature vector cannot silently redefine a subset.
MOTION_FEATURES = (
    "accel_sd",
    "accel_range",
    "accel_mean_jerk",
    "accel_max_jerk",
    "accel_still_fraction",
    "posture_x",
    "posture_y",
    "posture_z",
    "posture_change",
)
HEART_FEATURES = (
    "hr_mean",
    "hr_sd",
    "hr_min",
    "hr_max",
    "hr_vs_night_median",
    "hr_delta_prev",
)
TIME_FEATURES = ("elapsed_fraction",)

SUBSETS: Dict[str, Tuple[str, ...]] = {
    "heart_rate_only": HEART_FEATURES + TIME_FEATURES,
    "motion_only": MOTION_FEATURES + TIME_FEATURES,
    "motion_and_heart_rate": MOTION_FEATURES + HEART_FEATURES + TIME_FEATURES,
}


def subject_of(night_id: str) -> str:
    return night_id.split("/")[0]


def load_cached(cache_dir: Path) -> List[EpochSample]:
    """Every reduced night on disk, as epochs."""
    from datetime import datetime

    out: List[EpochSample] = []
    for path in sorted((cache_dir / "features").glob("*.json")):
        payload = json.loads(path.read_text())
        start = datetime.fromisoformat(payload["start"])
        for row in payload["epochs"]:
            out.append(
                EpochSample(
                    night_id=payload["night"],
                    index=row["index"],
                    start=start,
                    values=tuple(row["values"]),
                    label=row["label"],
                )
            )
    return out


def select(samples: Sequence[EpochSample], names: Sequence[str]) -> List[EpochSample]:
    """Keep only the named features, in FEATURE_NAMES order."""
    from dataclasses import replace

    keep = [i for i, n in enumerate(FEATURE_NAMES) if n in set(names)]
    return [replace(s, values=tuple(s.values[i] for i in keep)) for s in samples]


def leave_one_subject_out(
    samples: Sequence[EpochSample],
    implausible_above: float,
    seed: int = 0,
) -> CrossValidationSummary:
    """Train on every subject but one, score on the one, repeat."""
    from vitalgraph.ml.registry import require_sklearn

    require_sklearn()
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier

    by_subject: Dict[str, List[EpochSample]] = {}
    for s in samples:
        if s.label is not None:
            by_subject.setdefault(subject_of(s.night_id), []).append(s)

    results = {}
    for held_out in sorted(by_subject):
        train = [s for k, v in by_subject.items() if k != held_out for s in v]
        test = by_subject[held_out]
        if not train or not test:
            continue
        x = np.asarray([s.values for s in train], dtype=float)
        y = np.asarray([s.label for s in train], dtype=float)
        model = RandomForestClassifier(
            n_estimators=200,
            random_state=seed,
            class_weight="balanced",
            min_samples_leaf=3,
            n_jobs=-1,
        ).fit(x, y)
        predicted = [
            float(v)
            for v in model.predict(np.asarray([s.values for s in test], dtype=float))
        ]
        actual = [float(s.label) for s in test]
        results[held_out] = assess_skill(
            predicted, actual, [float(s.label) for s in train], implausible_above
        )
    return CrossValidationSummary(per_subject=results)


def run(samples: Sequence[EpochSample], task: str, implausible: float) -> Dict:
    out: Dict[str, object] = {}
    for name, names in SUBSETS.items():
        summary = leave_one_subject_out(select(samples, names), implausible)
        out[name] = {
            "n_features": len(names),
            "n_subjects": summary.n_subjects,
            "mean_accuracy": round(summary.mean_accuracy, 4),
            "mean_kappa": round(
                sum(a.kappa for a in summary.per_subject.values())
                / max(1, len(summary.per_subject)),
                4,
            ),
            "subjects_with_skill": sum(
                1 for a in summary.per_subject.values() if a.has_skill
            ),
            "subjects_without_skill": summary.subjects_without_skill,
            "implausible_subjects": summary.implausible_subjects,
            "worst": (
                {
                    "subject": summary.worst[0],
                    "accuracy": round(summary.worst[1].accuracy, 4),
                }
                if summary.worst
                else None
            ),
            "per_subject": {
                k: v.as_dict() for k, v in sorted(summary.per_subject.items())
            },
        }
        w = out[name]
        print(
            f"  {task:>10} · {name:<22} {w['mean_accuracy']:.3f} acc  "
            f"κ {w['mean_kappa']:+.3f}  {w['subjects_with_skill']}/{w['n_subjects']} "
            f"with skill",
            flush=True,
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "docs" / "bidsleep-evaluation.json"
    )
    args = parser.parse_args()

    samples = load_cached(args.cache)
    if not samples:
        raise SystemExit(f"no reduced nights in {args.cache}; run tools.fetch_bidsleep")

    subjects = sorted({subject_of(s.night_id) for s in samples})
    nights = sorted({s.night_id for s in samples})
    stages = Counter(
        SleepStage(int(s.label)).name.lower() for s in samples if s.label is not None
    )
    print(
        f"{len(samples)} epochs · {len(nights)} nights · {len(subjects)} subjects",
        flush=True,
    )

    if not args.no_context:
        # Smoothing is computed within a night, which the function enforces by
        # night_id -- so a subject's nights never bleed into one another.
        samples = add_temporal_context(samples)
        print(f"temporal context added: {len(samples[0].values)} features", flush=True)

    four = run(samples, "four-class", 0.90)
    two = run(
        collapse_to_sleep_wake(samples), "sleep/wake", IMPLAUSIBLE_SLEEP_WAKE_ACCURACY
    )

    payload = {
        "dataset": "physionet/bidsleep-dataset 1.0.0",
        "protocol": "leave-one-subject-out over whole subjects, never nights",
        "temporal_context": not args.no_context,
        "epochs": len(samples),
        "nights": len(nights),
        "subjects": len(subjects),
        "stage_counts": dict(stages),
        "four_class": four,
        "sleep_wake": two,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
