from __future__ import annotations

from collections.abc import Iterator

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pytest
from matplotlib.quiver import Quiver

from eeg_graph.config import Paths
from eeg_graph.nodes import Atlas
from eeg_graph.source import BrainMesh, fsaverage_mesh, load_dk_labels, source_atlas
from eeg_graph.viz import plot_brain_graph_3d, plot_brain_graph_views

_NAMES = ("occ-lh", "occ-rh", "front-lh", "front-rh")
_ATLAS = Atlas(
    space="source",
    names=_NAMES,
    hemispheres=("lh", "rh", "lh", "rh"),
    lobes=("occipital", "occipital", "frontal", "frontal"),
    coords_mm=np.array([[-40.0, -80.0, 10.0], [40.0, -80.0, 10.0], [-40.0, 30.0, 20.0], [40.0, 30.0, 20.0]]),
)
_MESH = BrainMesh(
    vertices_mm=np.array([[0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 50.0, 0.0], [0.0, 0.0, 50.0]]),
    triangles=np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]),
)
# poids croissants : (occ-rh -> front-rh), (occ-lh -> occ-rh), (occ-lh -> front-lh)
_EDGES = [("occ-rh", "front-rh", 0.4), ("occ-lh", "occ-rh", 0.6), ("occ-lh", "front-lh", 0.8)]


@pytest.fixture(autouse=True)
def _agg_backend() -> Iterator[None]:
    plt.switch_backend("Agg")
    yield
    plt.close("all")


def _graph(directed: bool) -> nx.Graph:
    g: nx.Graph = nx.DiGraph() if directed else nx.Graph()
    g.add_nodes_from(_NAMES)
    g.add_weighted_edges_from(_EDGES)
    return g


@pytest.mark.parametrize("directed", [False, True])
def test_brain_graph_3d_traces(directed: bool) -> None:
    fig = plot_brain_graph_3d(_graph(directed), _ATLAS, _MESH, n_weight_bins=3)
    types = [t.type for t in fig.data]
    assert types.count("mesh3d") == 1
    assert ("cone" in types) == directed
    lines = [t for t in fig.data if t.type == "scatter3d" and t.mode == "lines"]
    assert sum(int(np.isnan(np.asarray(t.x, dtype=float)).sum()) for t in lines) == len(_EDGES)  # 1 NaN / segment
    markers = [t for t in fig.data if t.type == "scatter3d" and t.mode == "markers"]
    assert sum(len(t.x) for t in markers) == len(_NAMES)
    assert [t.name for t in markers] == ["frontal (2)", "occipital (2)"]  # ordre de LOBE_ORDER


def test_brain_graph_3d_cones_point_to_target() -> None:
    fig = plot_brain_graph_3d(_graph(True), _ATLAS)
    cone = next(t for t in fig.data if t.type == "cone")
    u, v = np.asarray(cone.u), np.asarray(cone.v)
    assert u[1] == pytest.approx(1.0)  # occ-lh -> occ-rh : vers +x
    assert v[2] > 0.99  # occ-lh -> front-lh : vers l'avant (+y)


def test_brain_graph_3d_groups_and_sizes() -> None:
    fig = plot_brain_graph_3d(_graph(False), _ATLAS, groups=[0, 0, 1, 1], node_values=[1.0, 2.0, 3.0, 4.0])
    markers = {t.name: t for t in fig.data if t.mode == "markers"}
    assert set(markers) == {"communauté 0 (2)", "communauté 1 (2)"}
    assert max(markers["communauté 1 (2)"].marker.size) > max(markers["communauté 0 (2)"].marker.size)


@pytest.mark.parametrize("directed", [False, True])
def test_brain_graph_views(directed: bool) -> None:
    fig = plot_brain_graph_views(_graph(directed), _ATLAS, _MESH)
    assert len(fig.axes) == 3
    for ax in fig.axes:
        assert any(isinstance(c, Quiver) for c in ax.collections) == directed


_PATHS = Paths()


@pytest.mark.skipif(not (_PATHS.subjects_dir / "fsaverage" / "surf").exists(), reason="fsaverage absent")
def test_fsaverage_mesh_contains_dk_centroids() -> None:
    mesh = fsaverage_mesh(_PATHS)
    assert mesh.vertices_mm.shape == (2 * 10242, 3)
    assert mesh.triangles.shape == (2 * 20480, 3)
    assert mesh.triangles.min() == 0 and mesh.triangles.max() == mesh.vertices_mm.shape[0] - 1
    # même repère que les centroïdes : tous dans la boîte englobante de la surface piale
    centroids = source_atlas(load_dk_labels(_PATHS), _PATHS).coords_mm
    assert (centroids >= mesh.vertices_mm.min(axis=0)).all()
    assert (centroids <= mesh.vertices_mm.max(axis=0)).all()


@pytest.mark.parametrize("largest", [True, False])
def test_plot_region_values_order(largest: bool) -> None:
    from eeg_graph.viz import plot_region_values

    _, ax = plt.subplots()
    plot_region_values(ax, np.array([0.5, -2.0, 1.0, -1.0]), _ATLAS, "t", top=2, largest=largest, highlight=["occ-rh"])
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == (["front-lh", "occ-lh"] if largest else ["occ-rh", "front-rh"])
    bold = {t.get_text() for t in ax.get_yticklabels() if t.get_fontweight() == "bold"}
    assert bold == (set() if largest else {"occ-rh"})
