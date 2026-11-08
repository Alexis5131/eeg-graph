"""Contrôles de validation de la chaîne, sur EEGMMIDB R01 (yeux ouverts) / R02 (yeux fermés) comme jeu de test.

V1 : effet Berger en source (puissance alpha EC/EO maximale en pariéto-occipital) + bloc wPLI postérieur.
V2 : validité de tous les fichiers du cache.
V4 : temps de calcul par enregistrement.
V5 : équivalence du noyau W avec apply_inverse_epochs + extract_label_time_course.
V6 : sensibilité au choix de l'inverse (méthode × λ²).

Runs de tâche poing gauche / droit (R03/R07/R11 réel, R04/R08/R12 imaginé) :
ERD : désynchronisation alpha/bêta du cortex sensorimoteur pendant le mouvement réel et imaginé
(Pfurtscheller G., Lopes da Silva F. H. (1999), Clin Neurophysiol 110(11):1842-1857).
"""

from __future__ import annotations

from collections.abc import Sequence

import mne
import numpy as np
import pandas as pd
from mne.minimum_norm import apply_inverse_epochs
from numpy.typing import NDArray
from scipy.stats import hypergeom, spearmanr, wilcoxon

from eeg_graph.cache import RecordingResult, load_result, result_path, validate_connectivity
from eeg_graph.config import (
    REST_4S,
    TASK_LR,
    Band,
    EpochSpec,
    InverseConfig,
    NodeSpace,
    Paths,
    PipelineConfig,
    band_index,
)
from eeg_graph.connectivity import recording_connectivity
from eeg_graph.nodes import Atlas
from eeg_graph.pipeline import DATASET, PreparedSubject
from eeg_graph.preprocessing import epoch_onsets, epoch_ptp, ptp_keep_mask
from eeg_graph.source import compute_label_kernel, load_dk_labels, make_inverse, make_label_kernel, make_or_load_forward

POSTERIOR: tuple[str, ...] = ("parietal", "occipital")
# Cortex sensorimoteur (régions Desikan-Killiany), des deux côtés : gauche et droite sont regroupées.
MOTOR_ROI: tuple[str, ...] = tuple(
    f"{region}-{hemi}" for region in ("precentral", "postcentral", "paracentral") for hemi in ("lh", "rh")
)
TASK_CONDITIONS: tuple[str, ...] = ("mm_lr", "mi_lr")
TASK_LABELS: tuple[str, ...] = ("left", "right")
REST_LABEL = "rest"

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


# --- Runs de tâche : désynchronisation liée à l'événement (ERD) -------------------------------------------


def load_task_results(
    paths: Paths, subjects: Sequence[int], node_space: NodeSpace, runs: Sequence[int] = (3, 4, 7, 8, 11, 12)
) -> dict[int, dict[str, list[RecordingResult]]]:
    """Runs de tâche (découpage TASK_LR) présents dans le cache, groupés par sujet puis condition."""
    out: dict[int, dict[str, list[RecordingResult]]] = {}
    for s in subjects:
        for run in runs:
            path = result_path(paths.cache, DATASET, node_space, s, run, TASK_LR.name)
            if path.exists():
                r = load_result(path)
                out.setdefault(s, {}).setdefault(r.condition, []).append(r)
    return out


def epoch_label_counts(task_results: dict[int, dict[str, list[RecordingResult]]]) -> pd.DataFrame:
    """Epochs retenues par sujet, run et label (après rejet crête-à-crête)."""
    rows = [
        {"subject": s, "run": r.run, "condition": r.condition, "n_epochs_total": r.n_epochs_total}
        | {label: int((r.epoch_labels == label).sum()) for label in (REST_LABEL, *TASK_LABELS)}
        for s, by_cond in task_results.items()
        for results in by_cond.values()
        for r in results
    ]
    return pd.DataFrame(rows).sort_values(["subject", "run"], ignore_index=True)


def erd(results: Sequence[RecordingResult], band: str) -> NDArray[np.float64]:
    """ERD par région : moyenne des log10 P des epochs tâche (left + right) - moyenne des epochs rest.

    Les epochs des runs d'une même condition sont regroupées (même ICA et même noyau inverse pour un sujet).
    Valeur négative = désynchronisation ; 10**ERD - 1 donne la variation relative de puissance.
    Définition d'après Pfurtscheller & Lopes da Silva (1999), en log du rapport plutôt qu'en pourcentage.
    """
    b = results[0].bands.index(band)
    power = np.concatenate([r.power_epochs[:, b] for r in results]).astype(np.float64)
    labels = np.concatenate([r.epoch_labels for r in results])
    task = np.isin(labels, TASK_LABELS)
    rest = labels == REST_LABEL
    if not task.any() or not rest.any():
        raise ValueError(f"ERD {band} : il faut des epochs tâche et rest ({task.sum()} / {rest.sum()})")
    return power[task].mean(axis=0) - power[rest].mean(axis=0)


def erd_by_subject(
    task_results: dict[int, dict[str, list[RecordingResult]]],
    bands: Sequence[str] = ("alpha", "beta"),
    conditions: Sequence[str] = TASK_CONDITIONS,
) -> dict[tuple[str, str], NDArray[np.float64]]:
    """(condition, bande) -> ERD (n_sujets, n_régions), sujets ayant la condition, dans l'ordre croissant."""
    subjects = sorted(s for s, by_cond in task_results.items() if all(c in by_cond for c in conditions))
    return {
        (cond, band): np.stack([erd(task_results[s][cond], band) for s in subjects])
        for cond in conditions
        for band in bands
    }


def roi_top_check(
    values: NDArray[np.floating], atlas: Atlas, roi: Sequence[str] = MOTOR_ROI, top: int = 10
) -> dict[str, object]:
    """(a) Les régions de la ROI sont-elles parmi les `top` valeurs les plus négatives ?

    Rang 1 = région la plus négative. `passed` si la ROI y est surreprésentée par rapport au hasard
    (loi hypergéométrique : `top` régions tirées parmi n, dont len(roi) dans la ROI ; p < 0.05).
    """
    names = list(atlas.names)
    order = np.argsort(values)
    rank = {names[i]: k + 1 for k, i in enumerate(order)}
    in_top = [r for r in roi if rank[r] <= top]
    p = float(hypergeom.sf(len(in_top) - 1, len(names), len(roi), top))
    return {
        "top_regions": [names[i] for i in order[:top]],
        "roi_ranks": {r: rank[r] for r in roi},
        "n_roi_in_top": len(in_top),
        "n_roi": len(roi),
        "expected_by_chance": len(roi) * top / len(names),
        "hypergeom_p": p,
        "passed": p < 0.05,
    }


def roi_mean(values: NDArray[np.floating], atlas: Atlas, roi: Sequence[str] = MOTOR_ROI) -> NDArray[np.float64]:
    """Moyenne sur les régions de la ROI (dernier axe = régions)."""
    idx = [atlas.names.index(r) for r in roi]
    return np.asarray(values, dtype=np.float64)[..., idx].mean(axis=-1)


def roi_erd_negative_test(roi_erd: NDArray[np.floating]) -> dict[str, object]:
    """(b) Wilcoxon sur les sujets, unilatéral : ERD de la ROI < 0."""
    x = np.asarray(roi_erd, dtype=np.float64)
    p = float(wilcoxon(x, alternative="less").pvalue)
    return {
        "n_subjects": len(x),
        "mean_erd_log10": float(x.mean()),
        "median_erd_log10": float(np.median(x)),
        "mean_power_change_pct": float((10 ** x.mean() - 1) * 100),
        "n_subjects_negative": int((x < 0).sum()),
        "wilcoxon_p_less": p,
        "passed": p < 0.05,
    }


def mm_vs_mi_test(roi_mm: NDArray[np.floating], roi_mi: NDArray[np.floating]) -> dict[str, object]:
    """(c) ERD plus forte (plus négative) en mouvement réel qu'imaginé : Wilcoxon apparié sur les sujets."""
    mm, mi = np.asarray(roi_mm, dtype=np.float64), np.asarray(roi_mi, dtype=np.float64)
    return {
        "n_subjects": len(mm),
        "mean_erd_mm": float(mm.mean()),
        "mean_erd_mi": float(mi.mean()),
        "n_subjects_mm_stronger": int((mm < mi).sum()),
        "wilcoxon_p_less": float(wilcoxon(mm, mi, alternative="less").pvalue),
        "wilcoxon_p_two_sided": float(wilcoxon(mm, mi).pvalue),
    }


def task_erd_report(
    erds: dict[tuple[str, str], NDArray[np.float64]], atlas: Atlas, roi: Sequence[str] = MOTOR_ROI, top: int = 10
) -> dict[str, object]:
    """Contrôles (a), (b) par condition et bande, (c) par bande, sur les ERD par sujet."""
    report: dict[str, object] = {"roi": list(roi)}
    for (cond, band), values in erds.items():
        report[f"{cond}_{band}"] = {
            "a_roi_in_top": roi_top_check(values.mean(axis=0), atlas, roi, top),
            "b_roi_erd_negative": roi_erd_negative_test(roi_mean(values, atlas, roi)),
        }
    for band in sorted({b for _, b in erds}, key=[b for _, b in erds].index):
        if ("mm_lr", band) in erds and ("mi_lr", band) in erds:
            report[f"c_mm_vs_mi_{band}"] = mm_vs_mi_test(
                roi_mean(erds["mm_lr", band], atlas, roi), roi_mean(erds["mi_lr", band], atlas, roi)
            )
    return report
