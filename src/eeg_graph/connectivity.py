"""Connectivité fonctionnelle par bande à partir du signal analytique (filtre FIR -> Hilbert).

Chaîne : séries continues des nœuds (n_nodes, T) -> filtre passe-bande FIR zero-phase sur le signal
continu (pas d'effet de bord par epoch) -> transformée de Hilbert -> découpage aux onsets retenus ->
une matrice (n_nodes, n_nodes) par epoch et par métrique.

Références :
- wPLI : Vinck et al. (2011), NeuroImage 55(4), 1548-1565.
- ciPLV : Bruña, Maestú, Pereda (2018), J Neural Eng 15(5), 056011.
- AEC orthogonalisée : Hipp et al. (2012), Nat Neurosci 15(6), 884-890 ; implémentation mne-connectivity.
"""

from __future__ import annotations

import logging

import mne
import numpy as np
from mne_connectivity import envelope_correlation
from numpy.typing import NDArray
from scipy.signal import hilbert

from eeg_graph.config import BANDS, CONN_METRICS, Band, ConnMetric

logger = logging.getLogger(__name__)

type FloatArray = NDArray[np.floating]
type ComplexArray = NDArray[np.complexfloating]

# Les séries de sources valent ~1e-11 (unités de l'inverse) : aucun seuil absolu dans les calculs,
# les métriques de phase sont invariantes à l'échelle de chaque nœud (cf. test de non-régression).
_TINY = np.finfo(np.float64).tiny


def band_analytic(y: FloatArray, sfreq: float, band: Band) -> ComplexArray:
    """Signal analytique des séries continues `y` (n_nodes, T) filtrées dans `band`."""
    filtered = mne.filter.filter_data(
        np.asarray(y, dtype=np.float64),
        sfreq,
        l_freq=band.fmin,
        h_freq=band.fmax,
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose="error",
    )
    return hilbert(filtered, axis=-1)


def slice_epochs(z: NDArray, onsets: NDArray[np.integer], n_times: int) -> NDArray:
    """Découpe (n_nodes, T) en (n_epochs, n_nodes, n_times) aux échantillons `onsets`."""
    onsets = np.asarray(onsets, dtype=int)
    if onsets.size and (onsets.min() < 0 or onsets.max() + n_times > z.shape[-1]):
        raise ValueError("Un onset sort du signal")
    idx = onsets[:, None] + np.arange(n_times)[None, :]  # (n_epochs, n_times)
    return np.moveaxis(z[:, idx], 0, 1)


def _zero_diag(c: FloatArray) -> FloatArray:
    n = c.shape[-1]
    c[..., np.arange(n), np.arange(n)] = 0.0
    return c


def wpli(z: ComplexArray) -> FloatArray:
    """wPLI par epoch, estimé sur les échantillons temporels de chaque epoch.

    z : (n_epochs, n_nodes, n_times) complexe. Retour : (n_epochs, n_nodes, n_nodes) dans [0, 1].
    Avec z = x + iy, Im(z_i conj(z_j)) = y_i x_j - x_i y_j.
    """
    x, y = z.real, z.imag
    out = np.empty((z.shape[0], z.shape[1], z.shape[1]), dtype=np.float64)
    for e in range(z.shape[0]):
        # (n, n, T) : ~24 Mo pour 68 nœuds × 640 échantillons, d'où la boucle sur les epochs.
        im = y[e][:, None, :] * x[e][None, :, :] - x[e][:, None, :] * y[e][None, :, :]
        num = np.abs(im.sum(axis=-1))
        den = np.abs(im).sum(axis=-1)
        out[e] = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return _zero_diag(out)


def ciplv(z: ComplexArray) -> FloatArray:
    """ciPLV par epoch : |Im M| / sqrt(1 - (Re M)^2), M = <exp(i(phi_i - phi_j))>_t."""
    mag = np.abs(z)
    u = np.divide(z, mag, out=np.zeros_like(z), where=mag > 0)
    m = np.einsum("eit,ejt->eij", u, u.conj()) / z.shape[-1]
    denom = np.sqrt(np.clip(1.0 - m.real**2, 1e-12, None))  # m est sans dimension
    out = np.abs(m.imag) / denom
    return _zero_diag(np.clip(out, 0.0, 1.0))


def aec_orth(z: ComplexArray) -> FloatArray:
    """Corrélation d'enveloppe orthogonalisée par paire (mne-connectivity), par epoch."""
    conn = envelope_correlation(z, orthogonalize="pairwise", log=False, absolute=True, verbose=False)
    dense = np.asarray(conn.get_data(output="dense"), dtype=np.float64)
    dense = dense.reshape(z.shape[0], z.shape[1], z.shape[1])
    dense = 0.5 * (dense + np.swapaxes(dense, -1, -2))
    return _zero_diag(np.clip(dense, 0.0, 1.0))


def band_log_power(z: ComplexArray) -> FloatArray:
    """log10 de la puissance moyenne |z|^2 par epoch et par nœud : (n_epochs, n_nodes)."""
    return np.log10(np.maximum(np.mean(np.abs(z) ** 2, axis=-1), _TINY))


_METRIC_FUNCS = {"wpli": wpli, "ciplv": ciplv, "aec_orth": aec_orth}


def recording_connectivity(
    y: FloatArray,
    sfreq: float,
    onsets: NDArray[np.integer],
    n_times: int,
    bands: tuple[Band, ...] = BANDS,
    metrics: tuple[ConnMetric, ...] = CONN_METRICS,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Connectivité de toutes les epochs d'un enregistrement.

    y : (n_nodes, T) séries continues des nœuds. onsets : échantillons de début des epochs retenues.
    Retour : conn_epochs (n_epochs, n_metrics, n_bands, n_nodes, n_nodes) et
    power_epochs (n_epochs, n_bands, n_nodes), en float32.
    """
    n_nodes = y.shape[0]
    n_epochs = len(onsets)
    conn = np.zeros((n_epochs, len(metrics), len(bands), n_nodes, n_nodes), dtype=np.float32)
    power = np.zeros((n_epochs, len(bands), n_nodes), dtype=np.float32)
    for b, band in enumerate(bands):
        z = slice_epochs(band_analytic(y, sfreq, band), onsets, n_times)
        power[:, b] = band_log_power(z)
        for m, metric in enumerate(metrics):
            conn[:, m, b] = _METRIC_FUNCS[metric](z)
    return conn, power
