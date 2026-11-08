from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from eeg_graph.graph import to_networkx
from eeg_graph.metrics import (
    compute_metrics,
    laplacian_spectrum,
    louvain_consensus,
    mst_measures,
    n_zero_eigenvalues,
    partition_agreement,
    weighted_global_efficiency,
)


def _with_weights(g: nx.Graph, weight: float = 1.0) -> nx.Graph:
    for u, v in g.edges:
        g[u][v]["weight"] = weight
        g[u][v]["distance"] = 1.0 / weight
    return g


def test_path_graph_diameter_radius() -> None:
    m = compute_metrics(_with_weights(nx.path_graph(5)), n_louvain_runs=5)
    assert (m.diameter, m.radius) == (4, 2)
    assert m.is_connected and m.distances_on == "whole"


def test_disconnected_graph_convention() -> None:
    g = _with_weights(nx.disjoint_union(nx.complete_graph(3), nx.path_graph(4)))
    m = compute_metrics(g, n_louvain_runs=5)
    assert not m.is_connected
    assert m.n_components == 2
    assert m.distances_on == "largest_component"
    assert (m.diameter, m.radius) == (3, 2)
    assert m.n_zero_laplacian_eigs == 2
    assert np.isnan(m.eigenvector[:3]).all() and np.isfinite(m.eigenvector[3:]).all()


def test_complete_graph_efficiency_and_clustering() -> None:
    g = _with_weights(nx.complete_graph(6))
    assert nx.global_efficiency(g) == pytest.approx(1.0)
    assert weighted_global_efficiency(g) == pytest.approx(1.0)
    m = compute_metrics(g, n_louvain_runs=5)
    assert m.clustering_binary == pytest.approx(1.0)
    assert m.clustering_weighted == pytest.approx(1.0)


def test_weighted_efficiency_matches_binary_for_unit_weights() -> None:
    g = _with_weights(nx.karate_club_graph())
    assert weighted_global_efficiency(g) == pytest.approx(nx.global_efficiency(g))


def test_weighted_efficiency_hand_computed() -> None:
    # chemin a - b - c avec poids 0.5 et 0.25 : d_ab = 2, d_bc = 4, d_ac = 6
    w = np.array([[0, 0.5, 0], [0.5, 0, 0.25], [0, 0.25, 0]])
    expected = 2 * (1 / 2 + 1 / 4 + 1 / 6) / 6
    assert weighted_global_efficiency(to_networkx(w)) == pytest.approx(expected)


def test_laplacian_zero_eigenvalues_count_components() -> None:
    g = _with_weights(nx.disjoint_union_all([nx.cycle_graph(4), nx.path_graph(3), nx.complete_graph(5)]))
    eig = laplacian_spectrum(g, normalized=True)
    assert n_zero_eigenvalues(eig) == 3
    assert eig.min() > -1e-10 and eig.max() <= 2.0 + 1e-10


def test_louvain_recovers_two_cliques() -> None:
    g = _with_weights(nx.barbell_graph(6, 0))
    res = louvain_consensus(g, n_runs=10, seed=0)
    assert len(res.communities) == 2
    assert res.coassignment.shape == (12, 12)
    nodes = list(g.nodes)
    truth = [0] * 6 + [1] * 6
    assert partition_agreement(res.communities, nodes, truth)["ari"] == pytest.approx(1.0)


def test_mst_measures_star() -> None:
    m = mst_measures(nx.star_graph(5))
    assert m["mst_leaf_fraction"] == pytest.approx(5 / 6)
    assert m["mst_diameter"] == 2


def test_global_record_is_flat() -> None:
    record = compute_metrics(_with_weights(nx.cycle_graph(6)), n_louvain_runs=3).global_record()
    assert "betweenness" not in record and record["n_edges"] == 6
