"""Plancher du wPLI par epoch sous indépendance, comparé aux matrices de la cohorte MM / MI.

À lancer après scripts/mm_mi_cohort.py (ou sur un clone neuf : data/derived est versionné) :

    uv run python scripts/wpli_null_floor.py

Simulation : séries indépendantes (bruit blanc ou bruit à pic alpha), même filtre, même durée d'epoch (TASK_LR)
et même nombre d'epochs (médiane après égalisation) que l'analyse ; wPLI par epoch puis moyenne sur les epochs.
Sortie : reports/wpli_null_floor.json (plancher moyen ± écart-type sur les graines, valeurs observées).
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from eeg_graph.config import BANDS, TASK_LR, Paths, setup_logging
from eeg_graph.validation import NullNoise, wpli_null_floor

logger = logging.getLogger("wpli_null_floor")
SFREQ = 160.0
NOISES: tuple[NullNoise, ...] = ("white", "alpha_peak")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, default=5, help="nombre de simulations par (bande, bruit)")
    parser.add_argument("--n-nodes", type=int, default=20)
    args = parser.parse_args()
    setup_logging()
    paths = Paths()

    meta = pd.read_csv(paths.derived / "matrices_mm_mi.csv")
    n_epochs = int(meta.n_equalized.median())
    derived = np.load(paths.derived / "matrices_mm_mi.npz")
    bands = [str(b) for b in derived["bands"]]
    conditions = [str(c) for c in derived["conditions"]]
    iu = np.triu_indices(derived["matrices"].shape[-1], 1)

    band_records: dict[str, dict[str, object]] = {}
    for band in (b for b in BANDS if b.name in bands):
        bi = bands.index(band.name)
        edges = derived["matrices"][:, :, bi][..., iu[0], iu[1]]  # (sujets, conditions, arêtes)
        record: dict[str, object] = {
            "observed_mean": float(edges.mean()),
            "observed_group_p5_p95": {
                c: [float(np.percentile(edges[:, k].mean(0), q)) for q in (5, 95)] for k, c in enumerate(conditions)
            },
        }
        for noise in NOISES:
            floors = [
                wpli_null_floor(band, SFREQ, TASK_LR.duration, n_epochs, noise, args.n_nodes, seed)
                for seed in range(args.seeds)
            ]
            record[f"floor_{noise}"] = {"mean": float(np.mean(floors)), "sd": float(np.std(floors, ddof=1))}
            logger.info("%s, %s : plancher %.3f ± %.3f", band.name, noise, np.mean(floors), np.std(floors, ddof=1))
        logger.info("%s : observé %.3f", band.name, record["observed_mean"])
        band_records[band.name] = record

    out = {
        "epoch_s": TASK_LR.duration,
        "sfreq": SFREQ,
        "n_epochs": n_epochs,
        "n_nodes_simulated": args.n_nodes,
        "seeds": args.seeds,
        "bands": band_records,
    }
    target = paths.root / "reports" / "wpli_null_floor.json"
    target.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("Écrit %s", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
