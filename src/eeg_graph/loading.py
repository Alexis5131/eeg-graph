"""Chargement et contrôle des runs EEGMMIDB (PhysioNet, Schalk et al. 2004).

Seul module dépendant du dataset : il renvoie un `mne.io.Raw` avec montage, que le reste du pipeline
traite de façon générique.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from eeg_graph.config import EEGMMIDBSpec, Paths

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunCheck:
    subject: int
    run: int
    sfreq: float
    n_channels: int
    duration: float
    n_task_events: int
    median_task_event_duration: float
    reasons: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.reasons


class RunRejected(Exception):
    """Run exclu par les contrôles (fréquence, durée, annotations)."""

    def __init__(self, check: RunCheck) -> None:
        super().__init__(f"S{check.subject:03d}R{check.run:02d} rejeté : {'; '.join(check.reasons)}")
        self.check = check


def eegmmidb_file(subject: int, run: int, paths: Paths = Paths()) -> Path:
    """Chemin local de l'EDF, téléchargé depuis PhysioNet si absent."""
    files = mne.datasets.eegbci.load_data(
        subjects=subject, runs=[run], path=paths.eegmmidb, update_path=False, verbose="error"
    )
    return Path(files[0])


def load_eegmmidb_run(
    subject: int, run: int, paths: Paths = Paths(), montage: str = "fsaverage_1005"
) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_edf(eegmmidb_file(subject, run, paths), preload=True, verbose="error")
    mne.datasets.eegbci.standardize(raw)
    raw.set_montage(mne.channels.make_standard_montage(montage), on_missing="raise")
    raw.info["subject_info"] = {"his_id": f"S{subject:03d}"}
    return raw


def check_eegmmidb_run(raw: mne.io.BaseRaw, subject: int, run: int, spec: EEGMMIDBSpec = EEGMMIDBSpec()) -> RunCheck:
    """Contrôles par run. Les annotations des baselines sont ignorées (S089 est mal annoté).

    Un run de tâche tronqué (ex. S104 R08, 106 s) reste exploitable : simple avertissement.
    """
    reasons: list[str] = []
    warns: list[str] = []
    sfreq = float(raw.info["sfreq"])
    n_channels = len(mne.pick_types(raw.info, eeg=True))
    duration = raw.n_times / sfreq
    if not np.isclose(sfreq, spec.sfreq):
        reasons.append(f"sfreq {sfreq:g} Hz != {spec.sfreq:g} Hz")
    if n_channels != spec.n_channels:
        reasons.append(f"{n_channels} canaux EEG != {spec.n_channels}")

    n_task, median_task = 0, float("nan")
    if run in spec.baseline_runs:
        if duration < spec.min_baseline_duration:
            reasons.append(f"baseline de {duration:.1f} s < {spec.min_baseline_duration} s")
    else:
        if duration < spec.min_task_duration:
            warns.append(f"run de {duration:.1f} s < {spec.min_task_duration} s (tronqué)")
        task = [a for a in raw.annotations if a["description"] in ("T1", "T2")]
        n_task = len(task)
        if n_task == 0:
            reasons.append("aucune annotation T1/T2")
        else:
            median_task = float(np.median([a["duration"] for a in task]))
            if abs(median_task - spec.task_event_duration) > spec.task_event_tolerance:
                reasons.append(
                    f"essais de {median_task:.3f} s au lieu de {spec.task_event_duration} s (autre protocole)"
                )
    for w in warns:
        logger.warning("S%03dR%02d : %s", subject, run, w)
    return RunCheck(subject, run, sfreq, n_channels, duration, n_task, median_task, tuple(reasons), tuple(warns))
