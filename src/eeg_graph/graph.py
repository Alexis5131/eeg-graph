"""Matrice de connectivité -> graphe NetworkX pondéré, non orienté, simple.

Conventions :
- poids `weight` = connectivité dans [0, 1] ; longueur `distance` = 1 / weight (chemins pondérés) ;
- seuillage proportionnel : on garde les k = round(densité × n(n-1)/2) arêtes les plus fortes,
  ex aequo départagés par l'ordre du triangle supérieur (déterministe) ;
- union avec l'arbre couvrant maximal en option, pour garantir la connexité (diamètre défini).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray

type Matrix = NDArray[np.floating]


class GraphInputError(ValueError):
    """Matrice d'adjacence invalide (non carrée, non symétrique, valeurs hors bornes)."""


def check_adjacency(w: Matrix, atol: float = 1e-6) -> None:
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise GraphInputError(f"Matrice non carrée : {w.shape}")
    if not np.isfinite(w).all():
        raise GraphInputError("Matrice avec NaN ou inf")
    if not np.allclose(w, w.T, atol=atol):
        raise GraphInputError("Matrice non symétrique")
    if (w < -atol).any():
        raise GraphInputError("Poids négatifs")


def n_possible_edges(n_nodes: int) -> int:
    return n_nodes * (n_nodes - 1) // 2


def proportional_threshold(w: Matrix, density: float) -> Matrix:
    """Garde les arêtes les plus fortes pour atteindre `density` (fraction des paires)."""
    check_adjacency(w)
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density doit être dans ]0, 1] : {density}")
    n = w.shape[0]
    iu = np.triu_indices(n, k=1)
    values = w[iu]
    k = round(density * values.size)
    keep = np.argsort(-values, kind="stable")[:k]
    out = np.zeros_like(w, dtype=np.float64)
    out[iu[0][keep], iu[1][keep]] = values[keep]
    return out + out.T


def maximum_spanning_tree(w: Matrix) -> Matrix:
    """Arbre couvrant de poids maximal du graphe complet pondéré par `w`."""
    check_adjacency(w)
    g = nx.from_numpy_array(np.asarray(w, dtype=np.float64))
    tree = nx.maximum_spanning_tree(g, weight="weight", algorithm="kruskal")
    out = nx.to_numpy_array(tree, nodelist=range(w.shape[0]), weight="weight")
    return out


def mst_union(w_thresholded: Matrix, w_full: Matrix) -> Matrix:
    """Union du graphe seuillé et de l'arbre couvrant maximal du graphe complet (toujours connexe)."""
    tree = maximum_spanning_tree(w_full)
    return np.where(w_thresholded > 0, w_thresholded, tree)


def to_networkx(
    w: Matrix,
    node_names: Sequence[str] | None = None,
    node_attrs: Mapping[str, Sequence[object]] | None = None,
) -> nx.Graph:
    """Graphe NetworkX : une arête par poids > 0, attributs `weight` et `distance` = 1 / weight."""
    check_adjacency(w)
    n = w.shape[0]
    names = list(node_names) if node_names is not None else list(range(n))
    if len(names) != n:
        raise GraphInputError(f"{len(names)} noms pour {n} nœuds")
    g = nx.Graph()
    for i, name in enumerate(names):
        attrs = {key: values[i] for key, values in (node_attrs or {}).items()}
        g.add_node(name, **attrs)
    rows, cols = np.nonzero(np.triu(w, k=1))
    for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
        weight = float(w[r, c])
        g.add_edge(names[r], names[c], weight=weight, distance=1.0 / weight)
    return g


def build_graph(
    w: Matrix,
    density: float | None,
    node_names: Sequence[str] | None = None,
    node_attrs: Mapping[str, Sequence[object]] | None = None,
    ensure_connected: bool = False,
) -> nx.Graph:
    """Raccourci : seuillage proportionnel (None = graphe pondéré complet), MST en option, NetworkX."""
    adj: Matrix = np.asarray(w, dtype=np.float64)
    if density is not None:
        thresholded = proportional_threshold(adj, density)
        adj = mst_union(thresholded, adj) if ensure_connected else thresholded
    return to_networkx(adj, node_names, node_attrs)
