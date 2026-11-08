"""Construit le cache des matrices de connectivité EEGMMIDB pour une liste de sujets et de runs.

Exemple (cohorte complète, runs de tâche poing gauche / droit) :

    uv run python scripts/build_cache.py --subjects 1-109 --runs 3,4,7,8,11,12 --n-jobs 4

Toute la sortie (logs des workers compris) va dans logs/build_cache_<horodatage>.log ; le résumé par sujet
dans logs/build_cache_<horodatage>_summary.csv. Le calcul est idempotent : un sujet déjà en cache avec le
même hash de configuration est sauté.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime

from eeg_graph.config import PROJECT_ROOT, setup_logging
from eeg_graph.pipeline import build_cache

logger = logging.getLogger("build_cache")


def parse_int_list(text: str) -> list[int]:
    """'1-3,7,10-11' -> [1, 2, 3, 7, 10, 11]."""
    out: list[int] = []
    for part in text.split(","):
        start, _, end = part.strip().partition("-")
        out.extend(range(int(start), int(end or start) + 1))
    return sorted(set(out))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--subjects", type=parse_int_list, default=parse_int_list("1-109"), help="ex. 1-109 ou 1,5-8")
    parser.add_argument("--runs", type=parse_int_list, default=parse_int_list("3,4,7,8,11,12"), help="ex. 3,4")
    parser.add_argument("--n-jobs", type=int, default=4, help="workers joblib (~1.2 Go de RAM chacun)")
    parser.add_argument("--node-space", choices=("source", "sensor"), default="source")
    parser.add_argument("--overwrite", action="store_true", help="recalcule même si déjà en cache")
    args = parser.parse_args(argv)

    logs = PROJECT_ROOT / "logs"
    logs.mkdir(exist_ok=True)
    stem = logs / f"build_cache_{datetime.now():%Y%m%d_%H%M%S}"
    log_path = stem.with_suffix(".log")
    print(f"Logs : {log_path}", flush=True)
    # Redirection au niveau des descripteurs : les workers joblib (processus fils) héritent de stdout / stderr.
    log_file = log_path.open("a", buffering=1)
    os.dup2(log_file.fileno(), sys.stdout.fileno())
    os.dup2(log_file.fileno(), sys.stderr.fileno())

    setup_logging()
    logger.info(
        "Début : %d sujets (%d-%d), runs %s, %d workers, node_space=%s, overwrite=%s",
        len(args.subjects),
        args.subjects[0],
        args.subjects[-1],
        args.runs,
        args.n_jobs,
        args.node_space,
        args.overwrite,
    )
    t0 = time.perf_counter()
    summary = build_cache(args.subjects, args.runs, args.node_space, n_jobs=args.n_jobs, overwrite=args.overwrite)
    elapsed = time.perf_counter() - t0
    summary.to_csv(stem.with_name(stem.name + "_summary.csv"), index=False)
    failed = summary[summary.error.astype(bool)] if "error" in summary else summary.iloc[0:0]
    logger.info(
        "Fin : %d sujets en %.1f min, %d enregistrements, %d sujets en erreur %s",
        len(summary),
        elapsed / 60,
        int(summary.n_recordings.sum()),
        len(failed),
        failed.subject.tolist(),
    )
    return 1 if len(failed) else 0


if __name__ == "__main__":
    raise SystemExit(main())
