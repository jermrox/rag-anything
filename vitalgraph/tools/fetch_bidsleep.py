"""Fetch BIDSLEEP nights and reduce them to per-epoch features.

Downloads run concurrently because a night is 87 MB of motion and one at a
time would take most of a day. Each night is reduced and its raw motion
deleted as soon as it lands, so peak disk stays near the worker count times
90 MB rather than the dataset's 6 GB.

A night that fails does not stop the run. Losing one recording is a smaller
problem than losing an hour of downloads to an exception on the last file.

Usage::

    python -m tools.fetch_bidsleep                    # 1 night per subject
    python -m tools.fetch_bidsleep --nights 3         # up to 3 each
    python -m tools.fetch_bidsleep --workers 8
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

from vitalgraph.data.bidsleep import list_nights, list_subjects, load_night

ROOT = Path(__file__).resolve().parent.parent


def one(subject: str, night: int) -> Tuple[str, int, Dict[str, object] | None, str]:
    try:
        _, report = load_night(subject, night)
        return subject, night, report, ""
    except Exception as exc:  # noqa: BLE001 - one bad night must not end the run
        return subject, night, None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nights", type=int, default=1, help="nights per subject")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--subjects", type=int, default=0, help="0 = all")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "docs" / "bidsleep-fetch.json"
    )
    args = parser.parse_args()

    subjects = list_subjects()
    if args.subjects:
        subjects = subjects[: args.subjects]
    print(f"{len(subjects)} subjects", flush=True)

    jobs: List[Tuple[str, int]] = []
    for subject in subjects:
        try:
            nights = list_nights(subject)[: args.nights]
        except Exception as exc:  # noqa: BLE001
            print(f"{subject}: cannot list nights: {exc}", flush=True)
            continue
        jobs.extend((subject, n) for n in nights)

    print(f"{len(jobs)} nights to fetch", flush=True)
    started = time.time()
    done: List[Dict[str, object]] = []
    failed: List[Dict[str, str]] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, s, n) for s, n in jobs]
        for i, future in enumerate(as_completed(futures), 1):
            subject, night, report, error = future.result()
            elapsed = time.time() - started
            if report is None:
                failed.append({"night": f"{subject}/{night}", "error": error})
                print(f"[{i}/{len(jobs)}] {subject}/{night} FAILED {error}", flush=True)
            else:
                done.append(report)
                print(
                    f"[{i}/{len(jobs)}] {subject}/{night} "
                    f"{report['usable_epochs']}/{report['scored_epochs']} epochs "
                    f"({report['usable_fraction']:.0%}) · {elapsed / 60:.1f} min",
                    flush=True,
                )

    summary = {
        "nights_fetched": len(done),
        "nights_failed": len(failed),
        "total_usable_epochs": sum(int(r["usable_epochs"]) for r in done),
        "median_usable_fraction": (
            sorted(float(r["usable_fraction"]) for r in done)[len(done) // 2]
            if done
            else 0.0
        ),
        "minutes": round((time.time() - started) / 60, 1),
        "failures": failed,
        "per_night": done,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "per_night"}, indent=2))
    sys.exit(0 if done else 1)


if __name__ == "__main__":
    main()
