"""Contrôles de validation de la chaîne, sur EEGMMIDB R01 (yeux ouverts) / R02 (yeux fermés) comme jeu de test.

V1 : effet Berger en source (puissance alpha EC/EO maximale en pariéto-occipital) + bloc wPLI postérieur.
V2 : validité de tous les fichiers du cache.
V4 : temps de calcul par enregistrement.
V5 : équivalence du noyau W avec apply_inverse_epochs + extract_label_time_course.
V6 : sensibilité au choix de l'inverse (méthode × λ²).
"""

from __future__ import annotations

import mne
import numpy as np
import pandas as pd
from mne.minimum_norm import apply_inverse_epochs
from numpy.typing import NDArray
from scipy.stats import spearmanr, wilcoxon

from eeg_graph.cache import RecordingResult, load_result, result_path, validate_connectivity
from eeg_graph.config import REST_4S, Band, EpochSpec, InverseConfig, NodeSpace, Paths, PipelineConfig, band_index
from eeg_graph.connectivity import recording_connectivity
from eeg_graph.nodes import Atlas
from eeg_graph.pipeline import DATASET, PreparedSubject
from eeg_graph.preprocessing import epoch_onsets, epoch_ptp, ptp_keep_mask
from eeg_graph.source import compute_label_kernel, load_dk_labels, make_inverse, make_label_kernel, make_or_load_forward

POSTERIOR: tuple[str, ...] = ("parietal", "occipital")

type Pair = tuple[RecordingResult, RecordingResult]


def epoch_ptp_table(prepared: PreparedSubject, spec: EpochSpec, cfg: PipelineConfig) -> pd.DataFrame:
    """Amplitude crête-à-crête max par epoch, avant / après ICA, et décision de rejet (µV)."""
    rows = []
    for run, cleaned in prepared.cleaned.items():
        onsets, _, n_times = epoch_onsets(cleaned, spec)
        before = epoch_ptp(prepared.filtered[run].get_data(picks="eeg"), onsets, n_times)
        after_data = cleaned.get_data(picks="eeg")
        after = epoch_ptp(after_data, onsets, n_times)
        keep = ptp_keep_mask(after_data, onsets, n_times, cfg.preproc)
        rows += [
            {"run": run, "epoch": k, "ptp_before_uV": b * 1e6, "ptp_after_uV": a * 1e6, "kept": bool(kp)}
            for k, (b, a, kp) in enumerate(zip(before, after, keep, strict=True))
        ]
    return pd.DataFrame(rows)


def kernel_equivalence(raw: mne.io.BaseRaw, cfg: InverseConfig, paths: Paths, n_epochs: int = 3) -> float:
    """V5 : écart relatif max entre W @ X et la chaîne MNE complète sur quelques epochs de 4 s."""
    epochs = mne.make_fixed_length_epochs(raw, duration=4.0, preload=True, verbose="error")[:n_epochs]
    inv = make_inverse(raw.info, make_or_load_forward(raw.info, cfg, paths), cfg)
    labels = load_dk_labels(paths, cfg.parc)
    weights = compute_label_kernel(raw.info, inv, labels, cfg)
    stcs = apply_inverse_epochs(epochs, inv, lambda2=cfg.lambda2, method=cfg.method, pick_ori="normal", verbose="error")
    reference = np.stack(mne.extract_label_time_course(stcs, labels, inv["src"], mode=cfg.label_mode, verbose="error"))
    fast = np.einsum("lc,ect->elt", weights, epochs.get_data())
    return float(np.abs(fast - reference).max() / np.abs(reference).max())


def alpha_log_power_ratio(eo: RecordingResult, ec: RecordingResult) -> NDArray[np.float64]:
    """log10(P_EC / P_EO) en alpha, par nœud (moyenne des epochs)."""
    b = band_index("alpha")
    return (ec.power_epochs[:, b].mean(axis=0) - eo.power_epochs[:, b].mean(axis=0)).astype(np.float64)


def berger_check(ratio: NDArray[np.float64], atlas: Atlas) -> dict[str, object]:
    """V1 : la plus forte hausse alpha EC/EO est pariéto-occipitale, et le postérieur dépasse le reste."""
    posterior = np.isin(atlas.lobes, POSTERIOR)
    top10 = np.argsort(ratio)[::-1][:10]
    return {
        "top_region": atlas.names[int(top10[0])],
        "top10_posterior_fraction": float(posterior[top10].mean()),
        "posterior_minus_other_log_ratio": float(ratio[posterior].mean() - ratio[~posterior].mean()),
        "passed": bool(posterior[top10[0]] and ratio[posterior].mean() > ratio[~posterior].mean()),
    }


def posterior_block(w: NDArray[np.floating], atlas: Atlas) -> float:
    """Connectivité moyenne entre nœuds pariéto-occipitaux (triangle supérieur du bloc)."""
    post = np.isin(atlas.lobes, POSTERIOR)
    block = w[np.ix_(post, post)]
    return float(block[np.triu_indices(int(post.sum()), 1)].mean())


def load_rest_pairs(paths: Paths, subjects: list[int], node_space: NodeSpace) -> dict[int, Pair]:
    """Paires (R01, R02) découpées en 4 s présentes dans le cache."""
    pairs: dict[int, Pair] = {}
    for s in subjects:
        files = [result_path(paths.cache, DATASET, node_space, s, run, REST_4S.name) for run in (1, 2)]
        if all(f.exists() for f in files):
            pairs[s] = (load_result(files[0]), load_result(files[1]))
    return pairs


def wpli_posterior_block_test(pairs: dict[int, Pair], band: str = "alpha") -> dict[str, object]:
    """Bloc wPLI postérieur EC vs EO, test de Wilcoxon apparié (≥ 6 sujets)."""
    blocks = np.array(
        [
            [posterior_block(r.matrix("wpli", band), r.atlas()) for r in pair]  # (EO, EC)
            for pair in pairs.values()
        ]
    )
    stat = wilcoxon(blocks[:, 1], blocks[:, 0]) if len(blocks) >= 6 else None
    return {
        "n_subjects": len(blocks),
        "mean_EO": float(blocks[:, 0].mean()),
        "mean_EC": float(blocks[:, 1].mean()),
        "n_subjects_EC_gt_EO": int((blocks[:, 1] > blocks[:, 0]).sum()),
        "wilcoxon_p": float(stat.pvalue) if stat is not None else None,
    }


def validate_cache(paths: Paths) -> int:
    """V2 : relit et valide tous les .npz ; lève CacheValidationError au premier fichier invalide."""
    files = sorted((paths.cache / DATASET).rglob("*.npz"))
    for f in files:
        validate_connectivity(load_result(f).conn_epochs)
    return len(files)


def timing_summary(paths: Paths, node_space: NodeSpace, n_recordings: int = 530) -> dict[str, float]:
    """V4 : temps médian par enregistrement (index.csv) et extrapolation à `n_recordings`."""
    index = pd.read_csv(paths.cache / DATASET / "index.csv")
    ok = index[(index.status == "ok") & (index.node_space == node_space) & (index.epoch_spec == REST_4S.name)]
    median = float(ok.seconds.median())
    return {
        "median_seconds_per_recording": median,
        f"minutes_for_{n_recordings}_recordings": median * n_recordings / 60,
    }


def inverse_robustness(
    raw: mne.io.BaseRaw, cfg: PipelineConfig, paths: Paths, band: str = "alpha"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """V6 : grille méthode × λ² ; Spearman entre triangles supérieurs du wPLI et bloc postérieur par variante."""
    onsets, _, n_times = epoch_onsets(raw, REST_4S)
    onsets = onsets[ptp_keep_mask(raw.get_data(picks="eeg"), onsets, n_times, cfg.preproc)]
    bands: tuple[Band, ...] = (cfg.bands[band_index(band, cfg.bands)],)
    vectors: dict[str, NDArray[np.float64]] = {}
    blocks: dict[str, float] = {}
    for method in ("eLORETA", "dSPM", "MNE"):
        for lambda2 in (1.0, 1 / 9, 1 / 100):
            kernel = make_label_kernel(raw.info, InverseConfig(method=method, lambda2=lambda2), paths)
            conn, _ = recording_connectivity(kernel.project(raw), raw.info["sfreq"], onsets, n_times, bands, ("wpli",))
            w = conn.mean(axis=0)[0, 0].astype(np.float64)
            name = f"{method} λ²={lambda2:.3g}"
            vectors[name] = w[np.triu_indices(w.shape[0], 1)]
            blocks[name] = posterior_block(w, kernel.atlas)
    names = list(vectors)
    corr = pd.DataFrame(
        [[float(spearmanr(vectors[a], vectors[b]).statistic) for b in names] for a in names], index=names, columns=names
    )
    return corr, pd.DataFrame({"variant": names, "posterior_block_wpli": [blocks[n] for n in names]})
