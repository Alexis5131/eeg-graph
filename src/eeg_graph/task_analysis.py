"""Runs de tâche poing gauche / droit : cohorte, puis matrices de connectivité par sujet et condition.

Conditions :
- MM (mouvement réel) : epochs left + right des runs mm_lr (R03, R07, R11) ;
- MI (mouvement imaginé) : epochs left + right des runs mi_lr (R04, R08, R12) ;
- REST : epochs rest (pauses T0) des 6 runs.

Par sujet, chaque condition est réduite au même nombre d'epochs n = min(n_MM, n_MI, n_REST), tirées au
hasard, puis coupée en deux moitiés aléatoires disjointes A et B de n // 2 epochs (référence split-half :
ressemblance d'une condition avec elle-même). Le tirage dépend de (seed, sujet), pas de l'ordre de calcul.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from eeg_graph.cache import RecordingResult, load_result, result_path
from eeg_graph.config import SEED, TASK_LR, NodeSpace, Paths
from eeg_graph.nodes import Atlas
from eeg_graph.pipeline import DATASET

logger = logging.getLogger(__name__)

CONDITIONS: tuple[str, ...] = ("MM", "MI", "REST")
CONDITION_RUNS: dict[str, tuple[int, ...]] = {"MM": (3, 7, 11), "MI": (4, 8, 12)}
TASK_RUNS: tuple[int, ...] = (3, 4, 7, 8, 11, 12)
TASK_LABELS: tuple[str, ...] = ("left", "right")
REST_LABEL = "rest"
# Condition EEGMMIDB (config.EEGMMIDBSpec.conditions) dont viennent les epochs tâche de chaque condition.
_TASK_SOURCE: dict[str, str] = {"MM": "mm_lr", "MI": "mi_lr"}


# --- Cohorte ------------------------------------------------------------------------------------------------


def load_epoch_labels(path: Path) -> NDArray[np.str_]:
    """Labels des epochs retenues d'un fichier du cache, sans charger les matrices."""
    with np.load(path, allow_pickle=False) as f:
        return np.asarray(f["epoch_labels"])


def cohort_table(
    index: pd.DataFrame,
    subjects: Sequence[int],
    paths: Paths,
    node_space: NodeSpace = "source",
    min_runs: int = 2,
    min_task_epochs: int = 20,
) -> pd.DataFrame:
    """Statut de chaque sujet pour l'analyse MM / MI à partir de index.csv et des labels du cache.

    Inclusion : au moins `min_runs` runs exploitables en MM et en MI, et au moins `min_task_epochs` epochs
    tâche (left + right) retenues par condition.
    """
    idx = index[(index.epoch_spec == TASK_LR.name) & (index.node_space == node_space) & index.run.isin(TASK_RUNS)]
    rows = []
    for s in subjects:
        sub = idx[idx.subject == s].set_index("run")
        row: dict[str, object] = {"subject": s}
        reasons: list[str] = []
        if sub.empty:
            reasons.append("absent de index.csv")
        n_rest = 0
        for cond, runs in CONDITION_RUNS.items():
            ok_runs = [r for r in runs if r in sub.index and sub.loc[r, "status"] == "ok"]
            n_task = 0
            for r in ok_runs:
                labels = load_epoch_labels(paths.root / str(sub.loc[r, "path"]))
                n_task += int(np.isin(labels, TASK_LABELS).sum())
                n_rest += int((labels == REST_LABEL).sum())
            row[f"{cond}_runs_ok"] = len(ok_runs)
            row[f"{cond}_task_epochs"] = n_task
            if not sub.empty and len(ok_runs) < min_runs:
                reasons.append(f"{cond} : {len(ok_runs)} run(s) exploitable(s) < {min_runs}")
            if not sub.empty and n_task < min_task_epochs:
                reasons.append(f"{cond} : {n_task} epochs tâche < {min_task_epochs}")
        row["REST_epochs"] = n_rest
        not_ok = sub[sub.status != "ok"]
        row["runs_not_ok"] = "; ".join(
            f"R{r:02d} {st}: {why}"
            for r, st, why in zip(not_ok.index, not_ok.status, not_ok.reason.fillna(""), strict=True)
        )
        row["included"] = not reasons
        row["exclusion_reason"] = "; ".join(reasons)
        rows.append(row)
    return pd.DataFrame(rows)


# --- Epochs par condition, égalisation, split-half -----------------------------------------------------------


@dataclass(frozen=True)
class ConditionEpochs:
    """Matrices par epoch d'un sujet pour une métrique : condition -> (n_epochs, n_bandes, n, n)."""

    subject: int
    metric: str
    bands: tuple[str, ...]
    conn: dict[str, NDArray[np.float32]]
    runs: dict[str, NDArray[np.int64]]  # run d'origine de chaque epoch (traçabilité)
    atlas: Atlas

    def counts(self) -> dict[str, int]:
        return {c: int(self.conn[c].shape[0]) for c in CONDITIONS}


def condition_epochs(
    results: Sequence[RecordingResult], metric: str = "wpli", bands: Sequence[str] = ("alpha", "beta")
) -> ConditionEpochs:
    """Regroupe les epochs des runs de tâche d'un sujet en MM, MI et REST."""
    if not results:
        raise ValueError("Aucun enregistrement")
    parts: dict[str, list[NDArray[np.float32]]] = {c: [] for c in CONDITIONS}
    run_parts: dict[str, list[NDArray[np.int64]]] = {c: [] for c in CONDITIONS}
    for r in results:
        b = [r.bands.index(band) for band in bands]
        conn = r.conn_epochs[:, r.metrics.index(metric)][:, b]  # (n_epochs, n_bandes, n, n)
        task = np.isin(r.epoch_labels, TASK_LABELS)
        rest = r.epoch_labels == REST_LABEL
        for cond, source in _TASK_SOURCE.items():
            if r.condition == source:
                parts[cond].append(conn[task])
                run_parts[cond].append(np.full(int(task.sum()), r.run, dtype=np.int64))
        parts["REST"].append(conn[rest])
        run_parts["REST"].append(np.full(int(rest.sum()), r.run, dtype=np.int64))
    n = results[0].conn_epochs.shape[-1]
    empty = np.zeros((0, len(bands), n, n), dtype=np.float32)
    return ConditionEpochs(
        subject=results[0].subject,
        metric=metric,
        bands=tuple(bands),
        conn={c: np.concatenate(parts[c]) if parts[c] else empty for c in CONDITIONS},
        runs={c: np.concatenate(run_parts[c]) if run_parts[c] else np.zeros(0, np.int64) for c in CONDITIONS},
        atlas=results[0].atlas(),
    )


@dataclass(frozen=True)
class EpochSelection:
    """Indices tirés pour une condition : `full` (n epochs), moitiés disjointes `half_a` et `half_b` (n // 2)."""

    full: NDArray[np.int64]
    half_a: NDArray[np.int64]
    half_b: NDArray[np.int64]


def subject_rng(subject: int, seed: int = SEED) -> np.random.Generator:
    """Générateur propre au sujet : tirages reproductibles quel que soit l'ordre des sujets."""
    return np.random.default_rng([seed, subject])


def _condition_key(cond: str) -> tuple[int, str]:
    """Ordre de tirage fixe (MM, MI, REST, puis les autres par nom), indépendant de l'ordre du Mapping."""
    return (CONDITIONS.index(cond) if cond in CONDITIONS else len(CONDITIONS), cond)


def equalize_and_split(counts: Mapping[str, int], rng: np.random.Generator) -> dict[str, EpochSelection]:
    """Même nombre d'epochs n = min(counts) dans chaque condition, puis split-half A / B de n // 2 epochs.

    Les moitiés sont tirées parmi les n epochs retenues (si n est impair, une epoch reste hors des moitiés).
    """
    n = min(counts.values())
    if n < 2:
        raise ValueError(f"Pas assez d'epochs pour égaliser et couper en deux : {dict(counts)}")
    half = n // 2
    out: dict[str, EpochSelection] = {}
    for cond in sorted(counts, key=_condition_key):
        full = np.sort(rng.choice(counts[cond], size=n, replace=False)).astype(np.int64)
        order = rng.permutation(full)
        out[cond] = EpochSelection(full, np.sort(order[:half]), np.sort(order[half : 2 * half]))
    return out


@dataclass(frozen=True)
class SubjectMatrices:
    subject: int
    matrices: NDArray[np.float32]  # (n_conditions, n_bandes, n, n) : moyenne des n epochs égalisées
    halves: NDArray[np.float32]  # (n_conditions, 2, n_bandes, n, n) : moyennes des moitiés A et B
    n_epochs: int
    n_half: int
    available: dict[str, int]


def subject_matrices(epochs: ConditionEpochs, seed: int = SEED) -> SubjectMatrices:
    """Matrices moyennes égalisées et moitiés split-half d'un sujet, conditions dans l'ordre de CONDITIONS."""
    sel = equalize_and_split(epochs.counts(), subject_rng(epochs.subject, seed))
    matrices = np.stack([epochs.conn[c][sel[c].full].mean(axis=0) for c in CONDITIONS])
    halves = np.stack(
        [
            np.stack([epochs.conn[c][sel[c].half_a].mean(axis=0), epochs.conn[c][sel[c].half_b].mean(axis=0)])
            for c in CONDITIONS
        ]
    )
    n = len(sel[CONDITIONS[0]].full)
    return SubjectMatrices(
        subject=epochs.subject,
        matrices=matrices.astype(np.float32),
        halves=halves.astype(np.float32),
        n_epochs=n,
        n_half=n // 2,
        available=epochs.counts(),
    )


# --- Cohorte -> fichier dérivé --------------------------------------------------------------------------------


def load_subject_task_results(
    paths: Paths, subject: int, node_space: NodeSpace = "source", runs: Sequence[int] = TASK_RUNS
) -> list[RecordingResult]:
    """Runs de tâche d'un sujet présents dans le cache."""
    files = [result_path(paths.cache, DATASET, node_space, subject, run, TASK_LR.name) for run in runs]
    return [load_result(f) for f in files if f.exists()]


def build_cohort_matrices(
    paths: Paths,
    subjects: Sequence[int],
    metric: str = "wpli",
    bands: Sequence[str] = ("alpha", "beta"),
    node_space: NodeSpace = "source",
    seed: int = SEED,
) -> tuple[list[SubjectMatrices], Atlas]:
    """Matrices égalisées de chaque sujet (chargement sujet par sujet pour limiter la mémoire)."""
    out: list[SubjectMatrices] = []
    atlas: Atlas | None = None
    for s in subjects:
        epochs = condition_epochs(load_subject_task_results(paths, s, node_space), metric, bands)
        atlas = atlas or epochs.atlas
        if epochs.atlas.names != atlas.names:
            raise ValueError(f"S{s:03d} : nœuds différents du premier sujet")
        out.append(subject_matrices(epochs, seed))
        logger.info("S%03d : n = %d epochs par condition (%s)", s, out[-1].n_epochs, out[-1].available)
    if atlas is None:
        raise ValueError("Aucun sujet")
    return out, atlas


def cohort_meta(items: Sequence[SubjectMatrices]) -> pd.DataFrame:
    """Une ligne par sujet : epochs disponibles par condition, n égalisé, taille des moitiés."""
    return pd.DataFrame(
        [
            {"subject": m.subject}
            | {f"n_{c}_available": m.available[c] for c in CONDITIONS}
            | {"n_equalized": m.n_epochs, "n_half": m.n_half}
            for m in items
        ]
    )


def save_cohort_matrices(
    path: Path, items: Sequence[SubjectMatrices], atlas: Atlas, metric: str, bands: Sequence[str], seed: int = SEED
) -> pd.DataFrame:
    """Écrit le .npz (matrices et moitiés empilées par sujet) et le CSV de métadonnées à côté. Renvoie le CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = cohort_meta(items)
    np.savez_compressed(
        path,
        subjects=np.array([m.subject for m in items], dtype=np.int64),
        conditions=np.array(CONDITIONS),
        bands=np.array(bands),
        metric=np.array(metric),
        node_space=np.array(atlas.space),
        seed=np.array(seed),
        matrices=np.stack([m.matrices for m in items]),  # (sujets, conditions, bandes, n, n)
        halves=np.stack([m.halves for m in items]),  # (sujets, conditions, 2, bandes, n, n)
        n_epochs=meta.n_equalized.to_numpy(np.int64),
        n_half=meta.n_half.to_numpy(np.int64),
        node_names=np.array(atlas.names),
        node_hemispheres=np.array(atlas.hemispheres),
        node_lobes=np.array(atlas.lobes),
        node_coords_mm=atlas.coords_mm,
    )
    meta.to_csv(path.with_suffix(".csv"), index=False)
    return meta


@dataclass(frozen=True)
class CohortMatrices:
    subjects: NDArray[np.int64]
    conditions: tuple[str, ...]
    bands: tuple[str, ...]
    metric: str
    matrices: NDArray[np.float32]
    halves: NDArray[np.float32]
    n_epochs: NDArray[np.int64]
    n_half: NDArray[np.int64]
    atlas: Atlas

    def matrix(self, subject: int, condition: str, band: str) -> NDArray[np.float32]:
        s = int(np.flatnonzero(self.subjects == subject)[0])
        return self.matrices[s, self.conditions.index(condition), self.bands.index(band)]


def load_cohort_matrices(path: Path) -> CohortMatrices:
    with np.load(path, allow_pickle=False) as f:
        return CohortMatrices(
            subjects=f["subjects"],
            conditions=tuple(f["conditions"].tolist()),
            bands=tuple(f["bands"].tolist()),
            metric=str(f["metric"]),
            matrices=f["matrices"],
            halves=f["halves"],
            n_epochs=f["n_epochs"],
            n_half=f["n_half"],
            atlas=Atlas(
                space="sensor" if str(f["node_space"]) == "sensor" else "source",
                names=tuple(f["node_names"].tolist()),
                hemispheres=tuple(f["node_hemispheres"].tolist()),
                lobes=tuple(f["node_lobes"].tolist()),
                coords_mm=f["node_coords_mm"],
            ),
        )
