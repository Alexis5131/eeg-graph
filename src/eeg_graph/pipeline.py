"""Orchestration : enregistrement(s) EEG -> matrices de connectivité en cache.

`process_recording` est générique (tout `mne.io.Raw` prétraité + une fonction de projection vers les
nœuds). Pour EEGMMIDB, les étapes sont exposées séparément pour le notebook :
`prepare_eegmmidb_subject` (chargement, contrôles, ICA) -> `make_node_model` -> `compute_subject_results`
(connectivité + cache) ; `process_eegmmidb_subject` les enchaîne et `build_cache` parallélise sur les sujets.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import version

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray

from eeg_graph.cache import RecordingResult, load_result, result_path, save_result, update_index
from eeg_graph.config import EEGMMIDBSpec, EpochSpec, NodeSpace, Paths, PipelineConfig, setup_logging
from eeg_graph.connectivity import recording_connectivity
from eeg_graph.loading import RunCheck, check_eegmmidb_run, load_eegmmidb_run
from eeg_graph.nodes import Atlas, sensor_atlas
from eeg_graph.preprocessing import (
    EpochingError,
    IcaResult,
    apply_ica,
    epoch_onsets,
    filter_raw,
    fit_ica,
    ptp_keep_mask,
    set_average_reference,
)
from eeg_graph.source import make_label_kernel

logger = logging.getLogger(__name__)

DATASET = "eegmmidb"

type Projector = Callable[[mne.io.BaseRaw], NDArray[np.float64]]


def package_versions() -> dict[str, str]:
    return {pkg: version(pkg) for pkg in ("mne", "mne-connectivity", "numpy", "scipy", "networkx")}


@dataclass(frozen=True)
class NodeModel:
    atlas: Atlas
    project: Projector


def make_node_model(info: mne.Info, node_space: NodeSpace, cfg: PipelineConfig, paths: Paths) -> NodeModel:
    if node_space == "source":
        kernel = make_label_kernel(info, cfg.inverse, paths)
        return NodeModel(kernel.atlas, kernel.project)
    atlas = sensor_atlas(info)
    return NodeModel(atlas, lambda raw: raw.get_data(picks=list(atlas.names)))


@dataclass(frozen=True)
class RecordingConnectivity:
    conn_epochs: NDArray[np.float32]
    power_epochs: NDArray[np.float32]
    onsets: NDArray[np.int64]
    labels: NDArray[np.str_]
    n_times: int
    n_epochs_total: int


def process_recording(
    raw: mne.io.BaseRaw, nodes: NodeModel, spec: EpochSpec, cfg: PipelineConfig
) -> RecordingConnectivity:
    """Raw prétraité -> epochs (rejet crête-à-crête sur les capteurs) -> connectivité des nœuds."""
    onsets, labels, n_times = epoch_onsets(raw, spec)
    keep = ptp_keep_mask(raw.get_data(picks="eeg"), onsets, n_times, cfg.preproc)
    if not keep.any():
        raise EpochingError(f"Toutes les epochs '{spec.name}' rejetées par le critère d'amplitude")
    y = nodes.project(raw)  # (n_nodes, T) continu : le filtrage par bande se fait avant découpage
    conn, power = recording_connectivity(y, raw.info["sfreq"], onsets[keep], n_times, cfg.bands, cfg.metrics)
    return RecordingConnectivity(conn, power, onsets[keep], labels[keep], n_times, len(onsets))


@dataclass(frozen=True)
class PreparedSubject:
    """Runs d'un sujet chargés, contrôlés, filtrés (référence moyenne) et nettoyés par une ICA commune."""

    subject: int
    checks: dict[int, RunCheck]
    filtered: dict[int, mne.io.BaseRaw]  # avant ICA
    cleaned: dict[int, mne.io.BaseRaw]  # après ICA
    ica: IcaResult
    seconds: float

    @property
    def info(self) -> mne.Info:
        return next(iter(self.cleaned.values())).info


class NoUsableRunError(RuntimeError):
    """Aucun run du sujet ne passe les contrôles."""


def prepare_eegmmidb_subject(
    subject: int, runs: Sequence[int], cfg: PipelineConfig = PipelineConfig(), paths: Paths = Paths()
) -> PreparedSubject:
    """Chargement, contrôles par run, filtre + référence moyenne, ICA commune aux runs retenus."""
    t0 = time.perf_counter()
    dataset_spec = EEGMMIDBSpec()
    checks: dict[int, RunCheck] = {}
    filtered: dict[int, mne.io.BaseRaw] = {}
    for run in runs:
        raw = load_eegmmidb_run(subject, run, paths, cfg.inverse.montage)
        checks[run] = check_eegmmidb_run(raw, subject, run, dataset_spec)
        if checks[run].ok:
            filtered[run] = set_average_reference(filter_raw(raw, cfg.preproc))
        else:
            logger.warning("S%03dR%02d exclu : %s", subject, run, "; ".join(checks[run].reasons))
    if not filtered:
        raise NoUsableRunError(f"S{subject:03d} : aucun run exploitable parmi {list(runs)}")
    ica = fit_ica(list(filtered.values()), cfg.preproc, cfg.seed)
    cleaned = {run: apply_ica(raw, ica) for run, raw in filtered.items()}
    return PreparedSubject(subject, checks, filtered, cleaned, ica, time.perf_counter() - t0)


def epoch_specs_for(run: int, extra_specs: Sequence[EpochSpec] = ()) -> list[EpochSpec]:
    """Découpage par défaut du run + découpages supplémentaires pour les runs de repos."""
    dataset_spec = EEGMMIDBSpec()
    specs = [dataset_spec.epoch_spec(run)]
    if run in dataset_spec.baseline_runs:
        specs.extend(extra_specs)
    return specs


def compute_subject_results(
    prepared: PreparedSubject,
    nodes: NodeModel,
    node_space: NodeSpace,
    extra_specs: Sequence[EpochSpec] = (),
    cfg: PipelineConfig = PipelineConfig(),
    paths: Paths = Paths(),
    save: bool = True,
) -> list[RecordingResult]:
    """Connectivité de chaque run retenu (et de chaque découpage) ; écrit le cache et l'index si `save`."""
    dataset_spec = EEGMMIDBSpec()
    index_path = paths.cache / DATASET / "index.csv"
    subject = prepared.subject
    if save:
        for run, check in prepared.checks.items():
            if not check.ok:
                for spec in epoch_specs_for(run, extra_specs):
                    row = _index_row(subject, run, node_space, spec.name, "excluded", "; ".join(check.reasons))
                    update_index(index_path, row)
    shared_seconds = prepared.seconds / len(prepared.cleaned)
    results: list[RecordingResult] = []
    for run, raw in prepared.cleaned.items():
        for spec in epoch_specs_for(run, extra_specs):
            t0 = time.perf_counter()
            try:
                rc = process_recording(raw, nodes, spec, cfg)
            except EpochingError as err:
                logger.warning("S%03dR%02d %s : %s", subject, run, spec.name, err)
                if save:
                    update_index(index_path, _index_row(subject, run, node_space, spec.name, "no_epochs", str(err)))
                continue
            result = RecordingResult(
                dataset=DATASET,
                subject=subject,
                run=run,
                condition=dataset_spec.conditions[run],
                node_space=node_space,
                epoch_spec=spec.name,
                sfreq=float(raw.info["sfreq"]),
                n_times=rc.n_times,
                node_names=nodes.atlas.names,
                node_hemispheres=nodes.atlas.hemispheres,
                node_lobes=nodes.atlas.lobes,
                node_coords_mm=nodes.atlas.coords_mm,
                bands=tuple(b.name for b in cfg.bands),
                metrics=cfg.metrics,
                conn_epochs=rc.conn_epochs,
                power_epochs=rc.power_epochs,
                epoch_onsets=rc.onsets,
                epoch_labels=rc.labels,
                n_epochs_total=rc.n_epochs_total,
                ica_excluded=prepared.ica.excluded,
                config_hash=cfg.hash(),
                versions=package_versions(),
                seconds=shared_seconds + time.perf_counter() - t0,
            )
            if save:
                path = result_path(paths.cache, DATASET, node_space, subject, run, spec.name)
                save_result(result, path)
                row = _index_row(subject, run, node_space, spec.name, "ok", "")
                row |= {
                    "condition": result.condition,
                    "n_epochs": result.n_epochs,
                    "n_epochs_total": result.n_epochs_total,
                    "n_ica_excluded": len(prepared.ica.excluded),
                    "seconds": round(result.seconds, 2),
                    "path": str(path.relative_to(paths.root)),
                }
                update_index(index_path, row)
            logger.info(
                "S%03dR%02d %s : %d/%d epochs, ICA -%d, %.1f s",
                subject,
                run,
                spec.name,
                result.n_epochs,
                result.n_epochs_total,
                len(prepared.ica.excluded),
                result.seconds,
            )
            results.append(result)
    return results


def process_eegmmidb_subject(
    subject: int,
    runs: Sequence[int],
    node_space: NodeSpace = "source",
    extra_specs: Sequence[EpochSpec] = (),
    cfg: PipelineConfig = PipelineConfig(),
    paths: Paths = Paths(),
    overwrite: bool = False,
) -> list[RecordingResult]:
    """Pipeline complet d'un sujet (préparation -> nœuds -> connectivité -> cache), idempotent."""
    targets = [
        result_path(paths.cache, DATASET, node_space, subject, run, spec.name)
        for run in runs
        for spec in epoch_specs_for(run, extra_specs)
    ]
    if not overwrite and all(p.exists() for p in targets):
        cached = [load_result(p) for p in targets]
        if all(r.config_hash == cfg.hash() for r in cached):
            logger.info("S%03d : déjà en cache", subject)
            return cached
    try:
        prepared = prepare_eegmmidb_subject(subject, runs, cfg, paths)
    except NoUsableRunError as err:
        logger.warning("%s", err)
        return []
    nodes = make_node_model(prepared.info, node_space, cfg, paths)
    return compute_subject_results(prepared, nodes, node_space, extra_specs, cfg, paths)


def _batch_worker(
    subject: int, runs: Sequence[int], node_space: NodeSpace, extra_specs: Sequence[EpochSpec], overwrite: bool
) -> dict[str, object]:
    setup_logging()
    t0 = time.perf_counter()
    results = process_eegmmidb_subject(subject, runs, node_space, extra_specs, overwrite=overwrite)
    return {"subject": subject, "n_recordings": len(results), "seconds": round(time.perf_counter() - t0, 1)}


def build_cache(
    subjects: Sequence[int],
    runs: Sequence[int],
    node_space: NodeSpace = "source",
    extra_specs: Sequence[EpochSpec] = (),
    n_jobs: int = 1,
    overwrite: bool = False,
) -> pd.DataFrame:
    """Batch multi-sujets (joblib ; 4 workers max sur 8 Go de RAM). Idempotent."""
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_batch_worker)(s, runs, node_space, extra_specs, overwrite) for s in subjects
    )
    return pd.DataFrame(rows)


def _index_row(subject: int, run: int, node_space: str, epoch_spec: str, status: str, reason: str) -> dict[str, object]:
    return {
        "dataset": DATASET,
        "node_space": node_space,
        "subject": subject,
        "run": run,
        "epoch_spec": epoch_spec,
        "status": status,
        "reason": reason,
    }
