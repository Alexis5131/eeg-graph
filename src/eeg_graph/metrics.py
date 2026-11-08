"""Métriques de graphe exigées par le projet (NetworkX) + spectre du Laplacien et communautés.

Conventions (à reporter dans le rapport) :
- diamètre / rayon : en nombre de sauts (graphe binaire) ; si le graphe n'est pas connexe, calculés sur
  la plus grande composante connexe et `distances_on = "largest_component"` ;
- betweenness / closeness : chemins pondérés, longueur d'arête = 1 / poids ;
- eigenvector : pondéré, sur la plus grande composante (NaN ailleurs) ;
- clustering pondéré : Onnela et al. (2005), tel qu'implémenté par `nx.clustering(weight=...)` ;
- efficacité globale pondérée : moyenne de 1 / d_ij (Latora & Marchiori 2001 ; Rubinov & Sporns 2010),
  implémentation maison car NetworkX ne fournit que la version binaire.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

logger = logging.getLogger(__name__)

_ZERO_EIG_TOL = 1e-8


def weighted_global_efficiency(g: nx.Graph, distance: str = "distance") -> float:
    """E_w = 1 / (n(n-1)) * sum_{i != j} 1 / d_ij, d_ij plus court chemin pondéré (0 si non relié)."""
    n = g.number_of_nodes()
    if n < 2:
        return 0.0
    total = 0.0
    for _, lengths in nx.all_pairs_dijkstra_path_length(g, weight=distance):
        total += sum(1.0 / d for d in lengths.values() if d > 0)
    return total / (n * (n - 1))


def laplacian_spectrum(g: nx.Graph, normalized: bool = True, weight: str = "weight") -> NDArray[np.float64]:
    """Valeurs propres (croissantes) du Laplacien pondéré, normalisé ou combinatoire."""
    nodes = list(g.nodes)
    if normalized:
        lap = nx.normalized_laplacian_matrix(g, nodelist=nodes, weight=weight)
    else:
        lap = nx.laplacian_matrix(g, nodelist=nodes, weight=weight)
    return np.linalg.eigvalsh(lap.toarray().astype(np.float64))


def n_zero_eigenvalues(eigenvalues: NDArray[np.float64], tol: float = _ZERO_EIG_TOL) -> int:
    """Multiplicité de la valeur propre 0 du Laplacien = nombre de composantes connexes."""
    return int(np.sum(np.abs(eigenvalues) < tol))


@dataclass(frozen=True)
class LouvainResult:
    communities: list[frozenset[object]]
    modularity: float
    modularity_runs: NDArray[np.float64]
    coassignment: NDArray[np.float64]  # (n, n) : fraction des runs où i et j sont ensemble


def louvain_consensus(g: nx.Graph, n_runs: int = 100, seed: int = 42, resolution: float = 1.0) -> LouvainResult:
    """Louvain répété sur `n_runs` seeds ; garde la partition de modularité maximale.

    Blondel et al. (2008) ; la matrice de co-assignation mesure la stabilité de la partition.
    """
    nodes = list(g.nodes)
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    coassign = np.zeros((n, n), dtype=np.float64)
    best: list[frozenset[object]] = []
    best_q = -np.inf
    qs = np.empty(n_runs, dtype=np.float64)
    for r in range(n_runs):
        parts = nx.community.louvain_communities(g, weight="weight", resolution=resolution, seed=seed + r)
        q = nx.community.modularity(g, parts, weight="weight", resolution=resolution)
        qs[r] = q
        for part in parts:
            idx = [index[node] for node in part]
            coassign[np.ix_(idx, idx)] += 1.0
        if q > best_q:
            best_q, best = q, [frozenset(p) for p in parts]
    return LouvainResult(best, float(best_q), qs, coassign / n_runs)


def partition_labels(communities: Sequence[frozenset[object]], nodes: Sequence[object]) -> NDArray[np.int64]:
    lookup = {node: c for c, part in enumerate(communities) for node in part}
    return np.array([lookup[node] for node in nodes], dtype=np.int64)


def partition_agreement(
    communities: Sequence[frozenset[object]], nodes: Sequence[object], reference: Sequence[object]
) -> dict[str, float]:
    """NMI et ARI entre la partition et une partition de référence (lobes, hémisphères)."""
    labels = partition_labels(communities, nodes)
    return {
        "nmi": float(normalized_mutual_info_score(reference, labels)),
        "ari": float(adjusted_rand_score(reference, labels)),
    }


def mst_measures(tree: nx.Graph) -> dict[str, float]:
    """Mesures classiques de l'arbre couvrant (Stam et al. 2014) : feuilles, diamètre, kappa."""
    degrees = np.array([d for _, d in tree.degree()], dtype=np.float64)
    n = tree.number_of_nodes()
    return {
        "mst_leaf_fraction": float(np.sum(degrees == 1) / n),
        "mst_diameter": float(nx.diameter(tree)),
        "mst_kappa": float(np.mean(degrees**2) / np.mean(degrees)),
    }


@dataclass(frozen=True)
class GraphMetrics:
    n_nodes: int
    n_edges: int
    density: float
    is_connected: bool
    n_components: int
    largest_component_size: int
    distances_on: str  # "whole" | "largest_component"
    diameter: int
    radius: int
    avg_degree: float
    avg_strength: float
    clustering_binary: float
    clustering_weighted: float
    global_efficiency_binary: float
    global_efficiency_weighted: float
    assortativity: float
    modularity: float
    n_communities: int
    algebraic_connectivity: float
    n_zero_laplacian_eigs: int
    nodes: list[object] = field(repr=False)
    degree: NDArray[np.float64] = field(repr=False)
    strength: NDArray[np.float64] = field(repr=False)
    betweenness: NDArray[np.float64] = field(repr=False)
    closeness: NDArray[np.float64] = field(repr=False)
    eigenvector: NDArray[np.float64] = field(repr=False)
    clustering_nodes: NDArray[np.float64] = field(repr=False)
    community: NDArray[np.int64] = field(repr=False)
    laplacian_eigenvalues: NDArray[np.float64] = field(repr=False)

    def global_record(self) -> dict[str, object]:
        """Scalaires seulement, pour une ligne de DataFrame."""
        return {k: v for k, v in asdict(self).items() if not isinstance(v, np.ndarray | list)}


def _eigenvector_on_largest_component(g: nx.Graph, nodes: list[object]) -> NDArray[np.float64]:
    largest = max(nx.connected_components(g), key=len)
    out = np.full(len(nodes), np.nan)
    if len(largest) < 2:
        return out
    centrality = nx.eigenvector_centrality_numpy(g.subgraph(largest), weight="weight")
    for i, node in enumerate(nodes):
        if node in centrality:
            out[i] = centrality[node]
    return out


def compute_metrics(g: nx.Graph, seed: int = 42, n_louvain_runs: int = 100) -> GraphMetrics:
    """Toutes les métriques globales et nodales d'un graphe pondéré non orienté."""
    nodes = list(g.nodes)
    n = len(nodes)
    components = list(nx.connected_components(g))
    connected = len(components) == 1
    largest = max(components, key=len)
    dist_graph = g if connected else g.subgraph(largest)

    if dist_graph.number_of_nodes() > 1:
        ecc = nx.eccentricity(dist_graph)
        diameter, radius = max(ecc.values()), min(ecc.values())
    else:
        diameter = radius = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assortativity = float(nx.degree_assortativity_coefficient(g)) if g.number_of_edges() > 1 else float("nan")

    louvain = louvain_consensus(g, n_runs=n_louvain_runs, seed=seed)
    norm_spectrum = laplacian_spectrum(g, normalized=True)
    comb_spectrum = laplacian_spectrum(g, normalized=False)

    def as_array(values: dict[object, float]) -> NDArray[np.float64]:
        return np.array([values[node] for node in nodes], dtype=np.float64)

    degree = as_array(dict(g.degree()))
    strength = as_array(dict(g.degree(weight="weight")))
    clustering_nodes = as_array(nx.clustering(g, weight="weight"))

    return GraphMetrics(
        n_nodes=n,
        n_edges=g.number_of_edges(),
        density=nx.density(g),
        is_connected=connected,
        n_components=len(components),
        largest_component_size=len(largest),
        distances_on="whole" if connected else "largest_component",
        diameter=int(diameter),
        radius=int(radius),
        avg_degree=float(degree.mean()),
        avg_strength=float(strength.mean()),
        clustering_binary=float(nx.average_clustering(g)),
        clustering_weighted=float(clustering_nodes.mean()),
        global_efficiency_binary=float(nx.global_efficiency(g)),
        global_efficiency_weighted=weighted_global_efficiency(g),
        assortativity=assortativity,
        modularity=louvain.modularity,
        n_communities=len(louvain.communities),
        # Laplacien semi-défini positif : un -1e-16 est un zéro numérique
        algebraic_connectivity=max(float(comb_spectrum[1]), 0.0) if n > 1 else 0.0,
        n_zero_laplacian_eigs=n_zero_eigenvalues(norm_spectrum),
        nodes=nodes,
        degree=degree,
        strength=strength,
        betweenness=as_array(nx.betweenness_centrality(g, weight="distance", normalized=True)),
        closeness=as_array(nx.closeness_centrality(g, distance="distance")),
        eigenvector=_eigenvector_on_largest_component(g, nodes),
        clustering_nodes=clustering_nodes,
        community=partition_labels(louvain.communities, nodes),
        laplacian_eigenvalues=norm_spectrum,
    )
