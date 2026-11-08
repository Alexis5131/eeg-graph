"""Runs de tâche poing gauche / droit : cohorte, puis matrices de connectivité par sujet et condition.

Conditions :
- MM (mouvement réel) : epochs left + right des runs mm_lr (R03, R07, R11) ;
- MI (mouvement imaginé) : epochs left + right des runs mi_lr (R04, R08, R12) ;
- REST : epochs rest (pauses T0) des 6 runs.

Par sujet, chaque condition est réduite au même nombre d'epochs n = min(n_MM, n_MI, n_REST), tirées au
hasard, puis coupée en deux moitiés aléatoires disjointes A et B de n // 2 epochs (référence split-half :
ressemblance d'une condition avec elle-même). Le tirage dépend de (seed, sujet), pas de l'ordre de calcul.

Analyse (notebook 03) : graphes de groupe (description), métriques globales par sujet et tests appariés
(Wilcoxon, correction de Holm, corrélation rang-bisériale), similarité entre conditions rapportée à la
similarité intra-condition à nombre d'epochs égal (n // 2), hubs, communautés, robustesse au seuil.
Références : Holm S. (1979), Scand J Stat 6(2):65-70 ; corrélation rang-bisériale appariée :
Kerby D. S. (2014), Compr Psychol 3:11.IT.3.1 ; méthodes reprises des articles, pas de code.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import NDArray
from scipy.stats import rankdata, spearmanr, wilcoxon
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from eeg_graph.cache import RecordingResult, load_result, result_path
from eeg_graph.config import SEED, TASK_LR, NodeSpace, Paths
from eeg_graph.graph import build_graph, mst_union, proportional_threshold, to_networkx
from eeg_graph.metrics import GraphMetrics, compute_metrics, louvain_consensus, weighted_global_efficiency
from eeg_graph.nodes import Atlas
from eeg_graph.pipeline import DATASET
from eeg_graph.validation import MOTOR_ROI

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
    power: dict[str, NDArray[np.float32]] | None = None  # condition -> (n_epochs, n_bandes, n) log10 P

    def counts(self) -> dict[str, int]:
        return {c: int(self.conn[c].shape[0]) for c in CONDITIONS}


def condition_epochs(
    results: Sequence[RecordingResult], metric: str = "wpli", bands: Sequence[str] = ("alpha", "beta")
) -> ConditionEpochs:
    """Regroupe les epochs des runs de tâche d'un sujet en MM, MI et REST."""
    if not results:
        raise ValueError("Aucun enregistrement")
    parts: dict[str, list[NDArray[np.float32]]] = {c: [] for c in CONDITIONS}
    power_parts: dict[str, list[NDArray[np.float32]]] = {c: [] for c in CONDITIONS}
    run_parts: dict[str, list[NDArray[np.int64]]] = {c: [] for c in CONDITIONS}
    for r in results:
        b = [r.bands.index(band) for band in bands]
        conn = r.conn_epochs[:, r.metrics.index(metric)][:, b]  # (n_epochs, n_bandes, n, n)
        power = r.power_epochs[:, b]  # (n_epochs, n_bandes, n)
        task = np.isin(r.epoch_labels, TASK_LABELS)
        rest = r.epoch_labels == REST_LABEL
        for cond, source in _TASK_SOURCE.items():
            if r.condition == source:
                parts[cond].append(conn[task])
                power_parts[cond].append(power[task])
                run_parts[cond].append(np.full(int(task.sum()), r.run, dtype=np.int64))
        parts["REST"].append(conn[rest])
        power_parts["REST"].append(power[rest])
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
        power={
            c: np.concatenate(power_parts[c]) if power_parts[c] else np.zeros((0, len(bands), n), np.float32)
            for c in CONDITIONS
        },
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
    power: NDArray[np.float32] | None = None  # (n_conditions, n_bandes, n) : log10 P moyen des n epochs


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
    power = (
        None
        if epochs.power is None
        else np.stack([epochs.power[c][sel[c].full].mean(axis=0) for c in CONDITIONS]).astype(np.float32)
    )
    return SubjectMatrices(
        subject=epochs.subject,
        matrices=matrices.astype(np.float32),
        halves=halves.astype(np.float32),
        n_epochs=n,
        n_half=n // 2,
        available=epochs.counts(),
        power=power,
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
        # log10 P moyen des epochs égalisées (mêmes epochs que les matrices) : (sujets, conditions, bandes, n)
        power=np.stack([m.power if m.power is not None else np.full(m.matrices.shape[:-1], np.nan) for m in items]),
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
    power: NDArray[np.float32]  # (sujets, conditions, bandes, n)

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
            power=f["power"],
            atlas=Atlas(
                space="sensor" if str(f["node_space"]) == "sensor" else "source",
                names=tuple(f["node_names"].tolist()),
                hemispheres=tuple(f["node_hemispheres"].tolist()),
                lobes=tuple(f["node_lobes"].tolist()),
                coords_mm=f["node_coords_mm"],
            ),
        )


# --- Analyse : graphes de groupe (partie A) --------------------------------------------------------------------

COMPARISONS: tuple[tuple[str, str], ...] = (("MM", "MI"), ("MM", "REST"), ("MI", "REST"))
GLOBAL_METRICS: tuple[str, ...] = ("global_efficiency_weighted", "clustering_weighted", "modularity")


def group_matrix(cohort: CohortMatrices, condition: str, band: str) -> NDArray[np.float64]:
    """Matrice moyenne de groupe (moyenne des matrices égalisées des sujets)."""
    c, b = cohort.conditions.index(condition), cohort.bands.index(band)
    return cohort.matrices[:, c, b].mean(axis=0).astype(np.float64)


@dataclass(frozen=True)
class GraphDescription:
    record: dict[str, object]
    metrics: GraphMetrics
    graph: nx.Graph


def describe_graph(
    w: NDArray[np.floating], atlas: Atlas, density: float, seed: int = SEED, n_louvain_runs: int = 100
) -> GraphDescription:
    """Description exigée d'un graphe seuillé : taille, connexité, distances, communautés.

    Diamètre et rayon (en sauts) sur la plus grande composante, puis sur l'union avec l'arbre couvrant
    maximal du graphe complet (toujours connexe).
    """
    w = np.asarray(w, dtype=np.float64)
    g = build_graph(w, density, atlas.names, atlas.node_attrs())
    m = compute_metrics(g, seed=seed, n_louvain_runs=n_louvain_runs)
    union = to_networkx(mst_union(proportional_threshold(w, density), w), atlas.names)
    ecc = nx.eccentricity(union)
    isolated = sorted(str(v) for v in nx.isolates(g))
    record: dict[str, object] = {
        "n_nodes": m.n_nodes,
        "n_edges": m.n_edges,
        "density": round(m.density, 4),
        "is_connected": m.is_connected,
        "n_components": m.n_components,
        "largest_component_size": m.largest_component_size,
        "isolated_nodes": ", ".join(isolated) if isolated else "",
        "diameter_largest_component": m.diameter,
        "radius_largest_component": m.radius,
        "n_edges_mst_union": union.number_of_edges(),
        "diameter_mst_union": max(ecc.values()),
        "radius_mst_union": min(ecc.values()),
        "modularity": m.modularity,
        "n_communities": m.n_communities,
        "nmi_lobes": float(normalized_mutual_info_score(atlas.lobes, m.community)),
    }
    return GraphDescription(record, m, g)


def top_regions(
    m: GraphMetrics, k: int = 10, measures: Sequence[str] = ("strength", "betweenness", "eigenvector")
) -> pd.DataFrame:
    """Les k régions de plus forte valeur pour chaque centralité : colonnes « mesure » et « mesure_valeur »."""
    out: dict[str, list[object]] = {"rang": list(range(1, k + 1))}
    for name in measures:
        values = np.nan_to_num(np.asarray(getattr(m, name), dtype=np.float64), nan=-np.inf)
        order = np.argsort(-values, kind="stable")[:k]
        out[name] = [m.nodes[i] for i in order]
        out[f"{name}_valeur"] = [float(values[i]) for i in order]
    return pd.DataFrame(out)


# --- Analyse : métriques par sujet (partie B) ----------------------------------------------------------------


def graph_global_metrics(
    w: NDArray[np.floating], density: float, seed: int = SEED, n_louvain_runs: int = 100
) -> dict[str, float]:
    """Efficacité globale pondérée, clustering pondéré moyen et modularité de Louvain d'un graphe seuillé."""
    g = build_graph(np.asarray(w, dtype=np.float64), density)
    clustering = nx.clustering(g, weight="weight")
    return {
        "global_efficiency_weighted": weighted_global_efficiency(g),
        "clustering_weighted": float(np.mean(list(clustering.values()))),
        "modularity": louvain_consensus(g, n_runs=n_louvain_runs, seed=seed).modularity,
    }


def node_strength(w: NDArray[np.floating], density: float) -> NDArray[np.float64]:
    """Strength nodale du graphe seuillé (somme des poids gardés)."""
    return np.asarray(proportional_threshold(np.asarray(w, dtype=np.float64), density).sum(axis=1), dtype=np.float64)


def _subject_rows(
    subject: int,
    mats: NDArray[np.floating],
    cohort_meta: tuple[tuple[str, ...], tuple[str, ...]],
    densities: Sequence[float],
    seed: int,
    n_louvain_runs: int,
) -> list[dict[str, object]]:
    conditions, bands = cohort_meta
    rows = []
    for c, cond in enumerate(conditions):
        for b, band in enumerate(bands):
            for d in densities:
                rows.append(
                    {"subject": subject, "condition": cond, "band": band, "density": d}
                    | graph_global_metrics(mats[c, b], d, seed, n_louvain_runs)
                )
    return rows


def subject_global_metrics(
    cohort: CohortMatrices,
    densities: Sequence[float] = (0.10, 0.15, 0.20),
    seed: int = SEED,
    n_louvain_runs: int = 100,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Métriques globales du graphe de chaque sujet × condition × bande × densité (format long)."""
    meta = (cohort.conditions, cohort.bands)
    chunks = Parallel(n_jobs=n_jobs)(
        delayed(_subject_rows)(int(s), cohort.matrices[i], meta, densities, seed, n_louvain_runs)
        for i, s in enumerate(cohort.subjects)
    )
    return pd.DataFrame([row for chunk in chunks for row in chunk])


def subject_strengths(cohort: CohortMatrices, density: float) -> NDArray[np.float64]:
    """Strength nodale de chaque sujet × condition × bande : (sujets, conditions, bandes, n)."""
    s, c, b, n, _ = cohort.matrices.shape
    out = np.empty((s, c, b, n))
    for i in range(s):
        for j in range(c):
            for k in range(b):
                out[i, j, k] = node_strength(cohort.matrices[i, j, k], density)
    return out


def rank_biserial(x: NDArray[np.floating], y: NDArray[np.floating]) -> float:
    """Corrélation rang-bisériale appariée (Kerby 2014) : (R+ - R-) / (R+ + R-) sur les différences x - y non nulles.

    > 0 : x tend à dépasser y. Même convention d'exclusion des zéros que le Wilcoxon par défaut de SciPy.
    """
    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def holm(pvalues: Sequence[float]) -> NDArray[np.float64]:
    """p ajustées de Holm-Bonferroni (Holm 1979), monotones, plafonnées à 1."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = p.size
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def paired_test(x: NDArray[np.floating], y: NDArray[np.floating]) -> dict[str, float]:
    """Wilcoxon apparié bilatéral x vs y, médianes et taille d'effet rang-bisériale."""
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    res = wilcoxon(x, y)
    return {
        "n": int(x.size),
        "median_a": float(np.median(x)),
        "median_b": float(np.median(y)),
        "median_diff": float(np.median(x - y)),
        "W": float(res.statistic),
        "p": float(res.pvalue),
        "r_rb": rank_biserial(x, y),
    }


def global_metric_tests(
    table: pd.DataFrame, metrics: Sequence[str] = GLOBAL_METRICS, comparisons: Sequence[tuple[str, str]] = COMPARISONS
) -> pd.DataFrame:
    """Point 5 : Wilcoxon apparié par métrique et comparaison.

    Correction de Holm sur les 9 tests (3 métriques × 3 comparaisons) de chaque (bande, densité).
    """
    rows = []
    for (band, density), sub in table.groupby(["band", "density"], sort=True):
        wide = {m: sub.pivot(index="subject", columns="condition", values=m) for m in metrics}
        block: list[dict[str, object]] = []
        pvals: list[float] = []
        for m in metrics:
            for a, b in comparisons:
                test = paired_test(wide[m][a].to_numpy(), wide[m][b].to_numpy())
                pvals.append(test["p"])
                block.append({"band": band, "density": density, "metric": m, "comparison": f"{a} vs {b}"} | test)
        for row, p_adj in zip(block, holm(pvals), strict=True):
            row["p_holm"] = float(p_adj)
        rows.extend(block)
    return pd.DataFrame(rows)


# --- Analyse : même réseau ? similarité entre et intra-condition (point 6) ----------------------------------

type Similarity = Callable[[NDArray[np.float64], NDArray[np.float64]], float]


def upper(w: NDArray[np.floating]) -> NDArray[np.float64]:
    return np.asarray(w, dtype=np.float64)[np.triu_indices(w.shape[0], 1)]


def edge_spearman(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Spearman entre les poids de toutes les arêtes (triangle supérieur des matrices pondérées complètes)."""
    return float(spearmanr(upper(a), upper(b)).statistic)


def strength_spearman(density: float) -> Similarity:
    """Spearman entre les profils de strength nodale des graphes seuillés à `density`."""

    def sim(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
        return float(spearmanr(node_strength(a, density), node_strength(b, density)).statistic)

    return sim


def split_half_similarity(
    halves: NDArray[np.floating], conditions: Sequence[str], a: str, b: str, sim: Similarity
) -> dict[str, float]:
    """Similarité entre conditions et plafond intra-condition, à nombre d'epochs égal (moitiés).

    halves : (n_conditions, 2, n, n) pour un sujet et une bande.
    entre = moyenne des 4 croisements a-A/b-A, a-A/b-B, a-B/b-A, a-B/b-B ;
    intra = moyenne de a-A/a-B et b-A/b-B ; ratio = entre / intra.
    """
    ha = np.asarray(halves[conditions.index(a)], dtype=np.float64)
    hb = np.asarray(halves[conditions.index(b)], dtype=np.float64)
    between = float(np.mean([sim(ha[i], hb[j]) for i in (0, 1) for j in (0, 1)]))
    within = float(np.mean([sim(ha[0], ha[1]), sim(hb[0], hb[1])]))
    return {"between": between, "within": within, "ratio": between / within if within != 0 else float("nan")}


def similarity_table(
    cohort: CohortMatrices,
    densities: Sequence[float] = (0.10, 0.15, 0.20),
    comparisons: Sequence[tuple[str, str]] = COMPARISONS,
) -> pd.DataFrame:
    """Similarités par sujet, bande et comparaison (format long).

    measure = "edges" (Spearman des arêtes, indépendant du seuil : density = NaN) ou "strength" (par densité).
    """
    rows = []
    for i, s in enumerate(cohort.subjects):
        for k, band in enumerate(cohort.bands):
            halves = cohort.halves[i, :, :, k]  # (conditions, 2, n, n)
            measures: list[tuple[str, float, Similarity]] = [("edges", float("nan"), edge_spearman)]
            measures += [("strength", d, strength_spearman(d)) for d in densities]
            for measure, d, sim in measures:
                for a, b in comparisons:
                    rows.append(
                        {"subject": int(s), "band": band, "density": d, "measure": measure, "comparison": f"{a} vs {b}"}
                        | split_half_similarity(halves, cohort.conditions, a, b, sim)
                    )
    return pd.DataFrame(rows)


def bootstrap_median_ci(
    x: NDArray[np.floating], n_boot: int = 5000, seed: int = SEED, level: float = 0.95
) -> tuple[float, float, float]:
    """Médiane et IC percentile par bootstrap sur les sujets (rééchantillonnage avec remise)."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    medians = np.median(x[rng.integers(0, x.size, size=(n_boot, x.size))], axis=1)
    lo, hi = np.quantile(medians, [(1 - level) / 2, (1 + level) / 2])
    return float(np.median(x)), float(lo), float(hi)


def similarity_summary(table: pd.DataFrame, n_boot: int = 5000, seed: int = SEED) -> pd.DataFrame:
    """Par (bande, densité, mesure, comparaison) : médianes, ratio entre / intra (IC 95 % bootstrap),
    Wilcoxon entre vs intra ; Holm sur les comparaisons de chaque (bande, densité, mesure)."""
    rows = []
    keys = ["band", "density", "measure"]
    for key, sub in table.groupby(keys, sort=True, dropna=False):
        block: list[dict[str, object]] = []
        pvals: list[float] = []
        for comp, g in sub.groupby("comparison", sort=False):
            med, lo, hi = bootstrap_median_ci(g.ratio.to_numpy(), n_boot, seed)
            test = paired_test(g.between.to_numpy(), g.within.to_numpy())
            pvals.append(test["p"])
            block.append(
                dict(zip(keys, key, strict=True))
                | {
                    "comparison": comp,
                    "n": test["n"],
                    "median_between": test["median_a"],
                    "median_within": test["median_b"],
                    "median_ratio": med,
                    "ratio_ci_low": lo,
                    "ratio_ci_high": hi,
                    "n_ratio_below_1": int((g.ratio < 1).sum()),
                    "W": test["W"],
                    "p": test["p"],
                    "r_rb": test["r_rb"],
                }
            )
        for row, p_adj in zip(block, holm(pvals), strict=True):
            row["p_holm"] = float(p_adj)
        rows.extend(block)
    return pd.DataFrame(rows)


# --- Analyse : hubs, ROI, communautés, confusion puissance (points 7, 8) ------------------------------------


def top_hubs(strength: NDArray[np.floating], atlas: Atlas, k: int = 7) -> list[str]:
    """Les k régions de plus forte strength (moyenne des sujets déjà faite par l'appelant)."""
    order = np.argsort(-np.asarray(strength, dtype=np.float64), kind="stable")[:k]
    return [atlas.names[i] for i in order]


def jaccard(a: Sequence[object], b: Sequence[object]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else float("nan")


def roi_values(values: NDArray[np.floating], atlas: Atlas, roi: Sequence[str] = MOTOR_ROI) -> NDArray[np.float64]:
    """Moyenne sur la ROI (dernier axe = régions)."""
    idx = [atlas.names.index(r) for r in roi]
    return np.asarray(values, dtype=np.float64)[..., idx].mean(axis=-1)


def sensorimotor_module(labels: NDArray[np.integer], atlas: Atlas, roi: Sequence[str] = MOTOR_ROI) -> list[str]:
    """Régions de la communauté qui contient le plus de régions de la ROI (ex aequo : la plus petite communauté)."""
    labels = np.asarray(labels)
    roi_idx = [atlas.names.index(r) for r in roi]
    candidates = np.unique(labels[roi_idx])
    best = max(candidates, key=lambda c: (int(np.sum(labels[roi_idx] == c)), -int(np.sum(labels == c))))
    return [atlas.names[i] for i in np.flatnonzero(labels == best)]


def partition_similarity(a: NDArray[np.integer], b: NDArray[np.integer]) -> dict[str, float]:
    return {"nmi": float(normalized_mutual_info_score(a, b)), "ari": float(adjusted_rand_score(a, b))}


def power_strength_confound(
    power: NDArray[np.floating],
    strength: NDArray[np.floating],
    atlas: Atlas,
    a: str = "MM",
    b: str = "MI",
    conditions: Sequence[str] = CONDITIONS,
) -> dict[str, float]:
    """Spearman sur les sujets entre Δ puissance de la ROI (a - b, log10) et Δ strength de la ROI (a - b).

    power, strength : (sujets, conditions, n) pour une bande.
    """
    ia, ib = conditions.index(a), conditions.index(b)
    dp = roi_values(power[:, ia] - power[:, ib], atlas)
    ds = roi_values(strength[:, ia] - strength[:, ib], atlas)
    res = spearmanr(dp, ds)
    return {
        "n": int(dp.size),
        "rho": float(res.statistic),
        "p": float(res.pvalue),
        "median_dpower": float(np.median(dp)),
        "median_dstrength": float(np.median(ds)),
    }


# --- Robustesse (partie C) -------------------------------------------------------------------------------------


def robustness_table(
    tests: pd.DataFrame, similarities: pd.DataFrame, reference_density: float = 0.15, alpha: float = 0.05
) -> pd.DataFrame:
    """Tableau résultat × variante (bande × densité).

    Cellule : sens de l'effet (signe de r_rb : « + » = a > b, ou entre > intra), p de Holm, et « tient » si
    même sens et même statut de significativité que la variante de référence (même bande, `reference_density`).
    Les similarités d'arêtes ne dépendent pas du seuil : leur ligne est recopiée sur les trois densités.
    """
    t = tests.assign(result=tests.metric + " : " + tests.comparison)
    sim = similarities.assign(
        result="similarité " + similarities.measure + " : " + similarities.comparison + " (entre vs intra)"
    )
    densities = sorted(t.density.unique())
    sim_rows = []
    for _, row in sim.iterrows():
        for d in densities if pd.isna(row.density) else [row.density]:
            sim_rows.append(row.to_dict() | {"density": d})
    rows = pd.concat(
        [
            t[["result", "band", "density", "r_rb", "p_holm"]],
            pd.DataFrame(sim_rows)[["result", "band", "density", "r_rb", "p_holm"]],
        ],
        ignore_index=True,
    )
    rows["sign"] = np.where(rows.r_rb > 0, "+", np.where(rows.r_rb < 0, "−", "0"))
    rows["significant"] = rows.p_holm < alpha
    ref = rows.loc[np.isclose(rows.density, reference_density), ["result", "band", "sign", "significant"]]
    rows = rows.merge(ref, on=["result", "band"], how="left", suffixes=("", "_ref"))
    rows["holds"] = (rows.sign == rows.sign_ref) & (rows.significant == rows.significant_ref)
    rows["cell"] = rows.apply(
        lambda r: f"{r.sign} p={r.p_holm:.2g}{' *' if r.significant else ''} {'✓' if r.holds else '✗'}", axis=1
    )
    rows["variant"] = rows.band + " " + (rows.density * 100).round().astype(int).astype(str) + " %"
    order = [f"{b} {round(d * 100)} %" for b in sorted(rows.band.unique()) for d in densities]
    return rows.pivot(index="result", columns="variant", values="cell")[order]
