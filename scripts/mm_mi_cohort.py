"""Cohorte MM / MI : inclusion des sujets, contrôle ERD sur les inclus, matrices wPLI égalisées par condition.

À lancer après scripts/build_cache.py :

    uv run python scripts/mm_mi_cohort.py

Sorties :
- reports/tables/cohort_mm_mi.csv : statut et epochs par sujet, raison d'exclusion ;
- reports/cohort_mm_mi.json : statuts par run, raisons d'exclusion, distribution des epochs ;
- reports/validation_task_cohort.json et reports/figures/validation_task_erd_cohort_*.png : ERD (tâche 1) ;
- data/derived/matrices_mm_mi.npz (+ .csv) : matrices égalisées et moitiés split-half ;
- data/derived/run_matrices_mm_mi.npz (+ .csv) : une matrice par run de tâche (epochs tâche égalisées entre runs).
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from eeg_graph import validation
from eeg_graph.config import TASK_LR, Paths, setup_logging
from eeg_graph.pipeline import DATASET
from eeg_graph.task_analysis import (
    build_cohort_matrices,
    build_run_matrices,
    cohort_table,
    save_cohort_matrices,
    save_run_matrices,
)
from eeg_graph.viz import plot_region_values, save_figure

logger = logging.getLogger("mm_mi_cohort")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--last", type=int, default=109)
    parser.add_argument("--min-runs", type=int, default=2)
    parser.add_argument("--min-task-epochs", type=int, default=20)
    args = parser.parse_args(argv)
    setup_logging()
    paths = Paths()
    reports, tables = paths.root / "reports", paths.root / "reports" / "tables"
    subjects = list(range(args.first, args.last + 1))
    t0 = time.perf_counter()

    # 1. Statuts par run et inclusion
    index = pd.read_csv(paths.cache / DATASET / "index.csv")
    task = index[(index.epoch_spec == TASK_LR.name) & (index.node_space == "source") & index.subject.isin(subjects)]
    cohort = cohort_table(index, subjects, paths, min_runs=args.min_runs, min_task_epochs=args.min_task_epochs)
    cohort.to_csv(tables / "cohort_mm_mi.csv", index=False)
    included = cohort.loc[cohort.included, "subject"].tolist()
    excluded = cohort.loc[~cohort.included, ["subject", "exclusion_reason", "runs_not_ok"]]
    not_ok = task[task.status != "ok"]
    summary = {
        "n_subjects": len(subjects),
        "n_included": len(included),
        "inclusion_rule": {
            "min_runs_per_condition": args.min_runs,
            "min_task_epochs_per_condition": args.min_task_epochs,
        },
        "run_status_counts": task.groupby("status").size().to_dict(),
        "run_status_by_run": task.pivot_table(
            index="run", columns="status", values="subject", aggfunc="count", fill_value=0
        )
        .astype(int)
        .to_dict(orient="index"),
        "runs_not_ok": not_ok[["subject", "run", "status", "reason"]].to_dict(orient="records"),
        "excluded_subjects": excluded.to_dict(orient="records"),
        "epochs_by_condition_included": cohort.loc[cohort.included, ["MM_task_epochs", "MI_task_epochs", "REST_epochs"]]
        .describe()
        .round(1)
        .to_dict(),
        "epochs_by_condition_all": cohort[["MM_task_epochs", "MI_task_epochs", "REST_epochs"]]
        .describe()
        .round(1)
        .to_dict(),
    }
    logger.info("%d sujets inclus sur %d", len(included), len(subjects))

    # 2. Contrôle ERD de la tâche 1 sur les inclus
    erds, atlas, erd_subjects = validation.cohort_erd(paths, included)
    erd_report = validation.task_erd_report(erds, atlas)
    for (cond, band), values in erds.items():
        fig, ax = plt.subplots(figsize=(6, 5.5))
        title = f"ERD {band} {cond} : log10 P tâche − rest, moyenne de {len(values)} sujets"
        plot_region_values(ax, values.mean(axis=0), atlas, title, top=20, largest=False, highlight=validation.MOTOR_ROI)
        ax.set_xlabel("log10 P tâche − log10 P rest (ROI en gras)")
        save_figure(fig, paths.figures / f"validation_task_erd_cohort_{cond}_{band}.png")
        plt.close(fig)
    (reports / "validation_task_cohort.json").write_text(
        json.dumps({"subjects": erd_subjects, "erd": erd_report}, indent=2, ensure_ascii=False)
    )

    # 3. Matrices wPLI égalisées par sujet et condition, moitiés split-half
    items, matrix_atlas = build_cohort_matrices(paths, included, metric="wpli", bands=("alpha", "beta"))
    meta = save_cohort_matrices(paths.derived / "matrices_mm_mi.npz", items, matrix_atlas, "wpli", ("alpha", "beta"))
    summary["n_equalized"] = meta.n_equalized.describe().round(1).to_dict()
    summary["n_half"] = meta.n_half.describe().round(1).to_dict()

    # 4. Matrices par run (plafond par run du notebook 03)
    run_items, run_dropped, run_atlas = build_run_matrices(paths, included, metric="wpli", bands=("alpha", "beta"))
    run_meta = save_run_matrices(
        paths.derived / "run_matrices_mm_mi.npz", run_items, run_atlas, "wpli", ("alpha", "beta")
    )
    summary["run_matrices"] = {
        "n_subjects": len(run_items),
        "dropped_subjects": run_dropped,
        "n_per_run": run_meta.n_per_run.describe().round(1).to_dict(),
    }
    summary["seconds"] = round(time.perf_counter() - t0, 1)
    (reports / "cohort_mm_mi.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    logger.info("Terminé en %.1f s", summary["seconds"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
