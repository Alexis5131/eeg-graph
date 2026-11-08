"""Résultats en cache -> graphes, tableaux de métriques et comparaison des communautés aux lobes."""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from eeg_graph.cache import RecordingResult
from eeg_graph.config import SEED
from eeg_graph.graph import build_graph
from eeg_graph.metrics import GraphMetrics, compute_metrics

# Colonnes exigées par les consignes du projet (taille, type, connexité, distances, communautés).
REQUIRED_COLUMNS: tuple[str, ...] = (
    "n_nodes",
    "n_edges",
    "density",
    "is_connected",
    "n_components",
    "distances_on",
    "diameter",
    "radius",
    "avg_strength",
    "clustering_weighted",
    "global_efficiency_weighted",
    "modularity",
    "n_communities",
    "algebraic_connectivity",
)


def graph_from_result(
    result: RecordingResult, metric: str, band: str, density: float | None, ensure_connected: bool = False
) -> nx.Graph:
    """Graphe pondéré d'un enregistrement ; `density=None` = graphe pondéré complet."""
    atlas = result.atlas()
    return build_graph(result.matrix(metric, band), density, atlas.names, atlas.node_attrs(), ensure_connected)


def metrics_table(
    results: Sequence[RecordingResult],
    metric: str,
    band: str,
    densities: Sequence[float | None],
    ensure_connected: bool = False,
    n_louvain_runs: int = 100,
) -> pd.DataFrame:
    """Une ligne par (enregistrement, densité) avec toutes les métriques globales."""
    rows = []
    for r in results:
        for density in densities:
            g = graph_from_result(r, metric, band, density, ensure_connected)
            m = compute_metrics(g, seed=SEED, n_louvain_runs=n_louvain_runs)
            rows.append(
                {
                    "subject": r.subject,
                    "run": r.run,
                    "condition": r.condition,
                    "epoch_spec": r.epoch_spec,
                    "threshold": "complet" if density is None else f"{round(density * 100)} %",
                    **m.global_record(),
                }
            )
    return pd.DataFrame(rows)


def node_table(result: RecordingResult, m: GraphMetrics) -> pd.DataFrame:
    """Centralités et communauté de chaque nœud, avec hémisphère et lobe."""
    atlas = result.atlas()
    return pd.DataFrame(
        {
            "node": atlas.names,
            "hemisphere": atlas.hemispheres,
            "lobe": atlas.lobes,
            "degree": m.degree,
            "strength": m.strength,
            "betweenness": m.betweenness,
            "closeness": m.closeness,
            "eigenvector": m.eigenvector,
            "clustering": m.clustering_nodes,
            "community": m.community,
        }
    )


def community_agreement(result: RecordingResult, m: GraphMetrics) -> dict[str, float]:
    """NMI / ARI entre la partition de Louvain et les lobes, puis les hémisphères."""
    atlas = result.atlas()
    labels = np.asarray(m.community)
    return {
        "nmi_lobes": float(normalized_mutual_info_score(atlas.lobes, labels)),
        "ari_lobes": float(adjusted_rand_score(atlas.lobes, labels)),
        "nmi_hemispheres": float(normalized_mutual_info_score(atlas.hemispheres, labels)),
        "ari_hemispheres": float(adjusted_rand_score(atlas.hemispheres, labels)),
    }
