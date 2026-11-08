from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from eeg_graph.graph import (
    GraphInputError,
    build_graph,
    maximum_spanning_tree,
    mst_union,
    n_possible_edges,
    proportional_threshold,
    to_networkx,
)


def _random_symmetric(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.random((n, n))
    w = np.triu(a, 1)
    return w + w.T


@pytest.mark.parametrize(("density", "expected"), [(0.10, 228), (0.15, 342), (0.20, 456)])
def test_proportional_threshold_edge_counts_68_nodes(density: float, expected: int) -> None:
    w = _random_symmetric(68)
    t = proportional_threshold(w, density)
    assert n_possible_edges(68) == 2278
    assert np.count_nonzero(np.triu(t, 1)) == expected
    np.testing.assert_array_equal(t, t.T)
    assert np.all(np.diag(t) == 0)


def test_proportional_threshold_keeps_strongest() -> None:
    w = _random_symmetric(20, seed=1)
    t = proportional_threshold(w, 0.1)
    kept = t[np.triu_indices(20, 1)]
    dropped = w[np.triu_indices(20, 1)][kept == 0]
    assert kept[kept > 0].min() >= dropped.max()


def test_mst_has_n_minus_one_edges_and_union_is_connected() -> None:
    w = _random_symmetric(30, seed=2)
    tree = maximum_spanning_tree(w)
    assert np.count_nonzero(np.triu(tree, 1)) == 29
    sparse = proportional_threshold(w, 0.02)
    assert not nx.is_connected(to_networkx(sparse))
    assert nx.is_connected(to_networkx(mst_union(sparse, w)))


def test_to_networkx_attributes() -> None:
    w = np.array([[0, 0.5, 0], [0.5, 0, 0.25], [0, 0.25, 0]])
    g = to_networkx(w, ["a", "b", "c"], {"lobe": ["x", "y", "z"]})
    assert g.number_of_edges() == 2
    assert g["b"]["c"]["distance"] == pytest.approx(4.0)
    assert g.nodes["a"]["lobe"] == "x"


def test_invalid_inputs() -> None:
    with pytest.raises(GraphInputError):
        to_networkx(np.array([[0, 1], [0, 0]]))
    with pytest.raises(GraphInputError):
        to_networkx(np.array([[0, np.nan], [np.nan, 0]]))
    with pytest.raises(ValueError):
        proportional_threshold(_random_symmetric(5), 0.0)


def test_build_graph_full_weighted() -> None:
    w = _random_symmetric(10, seed=3)
    assert build_graph(w, density=None).number_of_edges() == 45
