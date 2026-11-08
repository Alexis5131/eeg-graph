"""Constantes et objets de configuration du pipeline EEG -> graphe.

Toutes les valeurs numériques du pipeline vivent ici ; les autres modules les reçoivent en argument.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

SEED = 42

PROJECT_ROOT = Path(__file__).resolve().parents[2]

type NodeSpace = Literal["source", "sensor"]
type ConnMetric = Literal["wpli", "ciplv", "aec_orth"]

CONN_METRICS: tuple[ConnMetric, ...] = ("wpli", "ciplv", "aec_orth")


@dataclass(frozen=True)
class Paths:
    root: Path = PROJECT_ROOT

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def eegmmidb(self) -> Path:
        return self.data / "eegmmidb"

    @property
    def subjects_dir(self) -> Path:
        return self.data / "subjects"

    @property
    def derivatives(self) -> Path:
        """Forward, noyaux inverses : calculés une fois, réutilisés."""
        return self.data / "derivatives"

    @property
    def cache(self) -> Path:
        return self.data / "cache"

    @property
    def figures(self) -> Path:
        return self.root / "reports" / "figures"


@dataclass(frozen=True)
class Band:
    name: str
    fmin: float
    fmax: float


BANDS: tuple[Band, ...] = (
    Band("delta", 1.0, 4.0),
    Band("theta", 4.0, 8.0),
    Band("alpha", 8.0, 13.0),
    Band("beta", 13.0, 30.0),
    Band("gamma", 30.0, 45.0),
)


def band_index(name: str, bands: tuple[Band, ...] = BANDS) -> int:
    return [b.name for b in bands].index(name)


@dataclass(frozen=True)
class PreprocConfig:
    l_freq: float = 0.5
    h_freq: float = 45.0
    # MNE recommande d'ajuster l'ICA sur une copie filtrée à 1 Hz (tutoriel 40_artifact_correction_ica).
    ica_l_freq: float = 1.0
    ica_n_components: int = 20
    ica_method: Literal["picard", "fastica", "infomax"] = "picard"
    eog_proxy_channels: tuple[str, ...] = ("Fp1", "Fp2")
    eog_threshold: float = 3.0
    max_eog_components: int = 3
    # Rejet d'epochs relatif à l'enregistrement (z-score robuste de l'amplitude crête-à-crête max),
    # inspiré de FASTER (Nolan et al. 2010). Un seuil fixe de 150 µV rejette 100 % des epochs de S001 :
    # l'alpha yeux fermés atteint 400-500 µV crête-à-crête en occipital, et un seuil fixe plus haut
    # rejetterait davantage la condition yeux fermés (biais de condition).
    ptp_robust_z: float = 3.0
    ptp_absolute_max: float = 1e-3  # volts, plafond pour les artefacts grossiers


@dataclass(frozen=True)
class InverseConfig:
    method: Literal["eLORETA", "dSPM", "MNE", "sLORETA"] = "eLORETA"
    # SNR = 1 pour des epochs uniques (exemple mne-connectivity « mne_inverse_label_connectivity »).
    lambda2: float = 1.0
    loose: float = 0.2
    depth: float = 0.8
    parc: str = "aparc"
    label_mode: str = "mean_flip"
    mindist: float = 5.0
    montage: str = "fsaverage_1005"


@dataclass(frozen=True)
class EpochSpec:
    """Découpage d'un enregistrement en epochs.

    kind="fixed" : fenêtres contiguës sans chevauchement de `length` secondes, label `label`.
    kind="events" : fenêtres [onset + tmin, onset + tmax] autour des annotations listées dans `events`
    (paires description -> label).
    """

    name: str
    kind: Literal["fixed", "events"]
    length: float | None = None
    label: str = "rest"
    events: tuple[tuple[str, str], ...] = ()
    tmin: float = 0.0
    tmax: float = 0.0

    @property
    def duration(self) -> float:
        if self.kind == "fixed":
            if self.length is None:
                raise ValueError(f"EpochSpec {self.name!r}: length requis pour kind='fixed'")
            return self.length
        return self.tmax - self.tmin


REST_4S = EpochSpec(name="rest4s", kind="fixed", length=4.0, label="rest")
# Protocole de Fraschini et al. (Entropy 2021) : 5 epochs de 12 s.
REST_12S = EpochSpec(name="rest12s", kind="fixed", length=12.0, label="rest")
# Runs de tâche poing gauche / droit (réel R03/R07/R11, imaginé R04/R08/R12) : T1/T2 durent 4.1 s, T0 4.2 s.
# Fenêtre 0.5-4.0 s après l'onset : on évite la réponse visuelle à la consigne et le débordement sur la
# consigne suivante. Les pauses T0 de ces runs fournissent la condition de repos.
TASK_LR = EpochSpec(
    name="task_lr",
    kind="events",
    events=(("T0", "rest"), ("T1", "left"), ("T2", "right")),
    tmin=0.5,
    tmax=4.0,
)


@dataclass(frozen=True)
class EEGMMIDBSpec:
    """Caractéristiques attendues d'un run EEGMMIDB (vérifiées sur les en-têtes des 1 526 EDF)."""

    sfreq: float = 160.0
    n_channels: int = 64
    baseline_runs: tuple[int, ...] = (1, 2)
    min_baseline_duration: float = 59.0
    min_task_duration: float = 115.0
    task_event_duration: float = 4.1
    task_event_tolerance: float = 0.15
    conditions: dict[int, str] = field(
        default_factory=lambda: {
            1: "rest_eo",
            2: "rest_ec",
            3: "mm_lr",
            4: "mi_lr",
            5: "mm_ff",
            6: "mi_ff",
            7: "mm_lr",
            8: "mi_lr",
            9: "mm_ff",
            10: "mi_ff",
            11: "mm_lr",
            12: "mi_lr",
            13: "mm_ff",
            14: "mi_ff",
        }
    )

    def epoch_spec(self, run: int) -> EpochSpec:
        if run in self.baseline_runs:
            return REST_4S
        if self.conditions[run] in ("mm_lr", "mi_lr"):
            return TASK_LR
        raise ValueError(f"Pas d'EpochSpec par défaut pour le run {run} ({self.conditions[run]})")


@dataclass(frozen=True)
class PipelineConfig:
    preproc: PreprocConfig = PreprocConfig()
    inverse: InverseConfig = InverseConfig()
    bands: tuple[Band, ...] = BANDS
    metrics: tuple[ConnMetric, ...] = CONN_METRICS
    seed: int = SEED

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()[:12]


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,  # Jupyter installe déjà des handlers sur le logger racine
    )
    logging.getLogger("mne").setLevel(logging.WARNING)
