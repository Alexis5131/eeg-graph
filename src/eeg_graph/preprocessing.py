"""Prétraitement générique d'un `mne.io.Raw` : filtrage, référence moyenne, ICA, epochs, rejet.

ICA : procédure du tutoriel MNE « Repairing artifacts with ICA »
(https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html) :
ajustement sur une copie filtrée à 1 Hz, application au signal filtré à 0.5 Hz. Faute de canal EOG,
Fp1/Fp2 servent de proxy (`find_bads_eog` accepte un canal EEG).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import mne
import numpy as np
from numpy.typing import NDArray

from eeg_graph.config import EpochSpec, PreprocConfig

logger = logging.getLogger(__name__)


class EpochingError(ValueError):
    """Aucune epoch exploitable pour la spécification demandée."""


def filter_raw(raw: mne.io.BaseRaw, cfg: PreprocConfig) -> mne.io.BaseRaw:
    """Passe-bande FIR zero-phase. Pas de notch : le passe-bas à 45 Hz coupe déjà au-delà de 56 Hz."""
    return (
        raw.copy()
        .load_data()
        .filter(cfg.l_freq, cfg.h_freq, method="fir", phase="zero", fir_design="firwin", verbose="error")
    )


def set_average_reference(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Référence moyenne en projection (requise par la modélisation inverse), projecteur appliqué."""
    out = raw.copy()
    out.set_eeg_reference("average", projection=True, verbose="error")
    out.apply_proj(verbose="error")
    return out


@dataclass(frozen=True)
class IcaResult:
    ica: mne.preprocessing.ICA
    excluded: tuple[int, ...]
    eog_scores: NDArray[np.float64]  # max |score| sur les canaux proxy, par composante


def fit_ica(raws: Sequence[mne.io.BaseRaw], cfg: PreprocConfig, seed: int) -> IcaResult:
    """Une ICA par sujet, ajustée sur la concaténation de ses runs filtrée à `cfg.ica_l_freq`."""
    concat = mne.concatenate_raws([r.copy() for r in raws], verbose="error")
    concat.filter(cfg.ica_l_freq, None, method="fir", phase="zero", fir_design="firwin", verbose="error")
    ica = mne.preprocessing.ICA(
        n_components=cfg.ica_n_components,
        method=cfg.ica_method,
        random_state=seed,
        max_iter="auto",
        verbose="error",
    )
    ica.fit(concat, verbose="error")
    proxies = [ch for ch in cfg.eog_proxy_channels if ch in concat.ch_names]
    if not proxies:
        logger.warning("Aucun canal proxy EOG %s : pas de rejet oculaire", cfg.eog_proxy_channels)
        return IcaResult(ica, (), np.zeros(ica.n_components_))
    inds, scores = ica.find_bads_eog(concat, ch_name=proxies, threshold=cfg.eog_threshold, verbose="error")
    score_matrix = np.atleast_2d(np.abs(np.asarray(scores, dtype=np.float64)))
    combined = score_matrix.max(axis=0)
    ranked = sorted(inds, key=lambda i: -combined[i])[: cfg.max_eog_components]
    ica.exclude = list(ranked)
    logger.info("ICA : %d composantes, %d retirées %s", ica.n_components_, len(ranked), ranked)
    return IcaResult(ica, tuple(int(i) for i in ranked), combined)


def apply_ica(raw: mne.io.BaseRaw, result: IcaResult) -> mne.io.BaseRaw:
    return result.ica.apply(raw.copy(), exclude=list(result.excluded), verbose="error")


def epoch_onsets(raw: mne.io.BaseRaw, spec: EpochSpec) -> tuple[NDArray[np.int64], NDArray[np.str_], int]:
    """Échantillons de début des epochs, labels, et longueur d'une epoch en échantillons.

    Les indices sont relatifs au premier échantillon de `raw` (directement utilisables sur get_data()).
    """
    sfreq = raw.info["sfreq"]
    n_times = round(spec.duration * sfreq)
    if spec.kind == "fixed":
        n_epochs = raw.n_times // n_times
        onsets = np.arange(n_epochs, dtype=np.int64) * n_times
        labels = np.full(n_epochs, spec.label)
    else:
        mapping = dict(spec.events)
        present = {d for d in raw.annotations.description if d in mapping}
        if not present:
            raise EpochingError(f"Aucune annotation {sorted(mapping)} dans l'enregistrement")
        event_id = {d: k + 1 for k, d in enumerate(sorted(present))}
        events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose="error")
        code_to_label = {code: mapping[d] for d, code in event_id.items()}
        onsets = events[:, 0] - raw.first_samp + round(spec.tmin * sfreq)
        labels = np.array([code_to_label[c] for c in events[:, 2]])
    inside = (onsets >= 0) & (onsets + n_times <= raw.n_times)
    if not inside.any():
        raise EpochingError(f"Aucune epoch '{spec.name}' entièrement dans le signal")
    return onsets[inside], labels[inside], n_times


def epoch_ptp(data: NDArray[np.floating], onsets: NDArray[np.int64], n_times: int) -> NDArray[np.float64]:
    """Amplitude crête-à-crête maximale sur les canaux, par epoch : (n_epochs,).

    Adapté de `reject_amplitude_artifacts` (projet Sleep-Analysis, src/preprocessing.py:116,
    https://github.com/Alexis5131/Sleep-Analysis), généralisé au multicanal et aux onsets.
    """
    idx = onsets[:, None] + np.arange(n_times)[None, :]
    return np.ptp(data[:, idx], axis=-1).max(axis=0)


def ptp_keep_mask(
    data: NDArray[np.floating], onsets: NDArray[np.int64], n_times: int, cfg: PreprocConfig
) -> NDArray[np.bool_]:
    """Garde les epochs dont l'amplitude max n'est pas aberrante pour cet enregistrement.

    Rejet si z robuste = (ptp - médiane) / (1.4826 × MAD) > cfg.ptp_robust_z, ou si ptp > plafond absolu.
    """
    ptp = epoch_ptp(data, onsets, n_times)
    keep = ptp <= cfg.ptp_absolute_max
    med = np.median(ptp)
    mad = 1.4826 * np.median(np.abs(ptp - med))
    if mad > 0:
        keep &= (ptp - med) / mad <= cfg.ptp_robust_z
    return keep
