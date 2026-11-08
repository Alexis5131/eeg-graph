"""Figures : matrices ordonnées par lobe, graphe circulaire par lobe, valeurs par région, graphe dans le cerveau."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from numpy.typing import NDArray

from eeg_graph.nodes import LOBE_ORDER, Atlas
from eeg_graph.source import BrainMesh

LOBE_COLORS: dict[str, str] = {
    "frontal": "#4C72B0",
    "central": "#64B5CD",
    "cingulate": "#8172B2",
    "insula": "#CCB974",
    "temporal": "#55A868",
    "parietal": "#DD8452",
    "occipital": "#C44E52",
}


def save_figure(fig: Figure, path: Path, dpi: int = 200) -> None:
    """Adapté de `_save` (projet Sleep-Analysis, src/visualization.py:34)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _lobe_boundaries(atlas: Atlas, order: NDArray[np.int64]) -> list[int]:
    keys = [(atlas.hemispheres[i], atlas.lobes[i]) for i in order]
    return [k for k in range(1, len(keys)) if keys[k] != keys[k - 1]]


def plot_connectivity_matrix(
    ax: Axes, w: NDArray[np.floating], atlas: Atlas, title: str, vmax: float | None = None
) -> None:
    order = atlas.display_order()
    im = ax.imshow(w[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
    for b in _lobe_boundaries(atlas, order):
        ax.axhline(b - 0.5, color="white", lw=0.4)
        ax.axvline(b - 0.5, color="white", lw=0.4)
    # bandes de couleur des lobes le long des axes
    colors = [LOBE_COLORS[atlas.lobes[i]] for i in order]
    for k, c in enumerate(colors):
        ax.add_patch(plt.Rectangle((-3.5, k - 0.5), 2.5, 1, color=c, clip_on=False))
    n_lh = sum(h == "lh" for h in atlas.hemispheres)
    ax.set_xticks([n_lh / 2, n_lh + (atlas.n_nodes - n_lh) / 2], ["gauche", "droite"])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)


def circular_positions(atlas: Atlas) -> dict[str, tuple[float, float]]:
    """Hémisphère gauche sur le demi-cercle gauche, droit en miroir ; frontal en haut, occipital en bas."""
    order = atlas.display_order()
    pos: dict[str, tuple[float, float]] = {}
    for hemi, sign in (("lh", -1.0), ("mid", 0.0), ("rh", 1.0)):
        idx = [i for i in order if atlas.hemispheres[i] == hemi]
        if not idx:
            continue
        if hemi == "mid":
            for k, i in enumerate(idx):
                pos[atlas.names[i]] = (0.0, 0.8 - 1.6 * k / max(len(idx) - 1, 1))
            continue
        angles = np.linspace(np.pi * 0.47, -np.pi * 0.47, len(idx))
        for theta, i in zip(angles, idx, strict=True):
            pos[atlas.names[i]] = (sign * np.cos(theta), np.sin(theta))
    return pos


def plot_circular_graph(
    ax: Axes,
    g: nx.Graph,
    atlas: Atlas,
    community: Sequence[int] | NDArray[np.integer],
    betweenness: Sequence[float] | NDArray[np.floating],
    title: str,
) -> None:
    """Couleur = communauté (Louvain), taille = betweenness, épaisseur = poids."""
    pos = circular_positions(atlas)
    nodes = list(g.nodes)
    weights = np.array([g[u][v]["weight"] for u, v in g.edges]) if g.number_of_edges() else np.array([1.0])
    widths = 0.3 + 2.5 * (weights - weights.min()) / max(np.ptp(weights), 1e-12)
    nx.draw_networkx_edges(g, pos, ax=ax, width=widths, alpha=0.35, edge_color="0.3")
    bet = np.asarray(betweenness, dtype=np.float64)
    sizes = 20 + 400 * bet / max(bet.max(), 1e-12)
    cmap = plt.get_cmap("tab10")
    nx.draw_networkx_nodes(
        g,
        pos,
        nodelist=nodes,
        ax=ax,
        node_size=sizes,
        node_color=[cmap(int(c) % 10) for c in community],
        edgecolors="black",
        linewidths=0.4,
    )
    for name, (x, y) in pos.items():
        label = name.rsplit("-", 1)[0] if atlas.space == "source" else name
        ax.text(1.12 * x, 1.12 * y, label, fontsize=4.5, ha="left" if x > 0 else "right", va="center")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_region_values(ax: Axes, values: NDArray[np.floating], atlas: Atlas, title: str, top: int = 20) -> None:
    """Barres horizontales des `top` plus grandes valeurs, colorées par lobe."""
    order = np.argsort(values)[::-1][:top]
    ax.barh(range(len(order)), values[order], color=[LOBE_COLORS[atlas.lobes[i]] for i in order])
    ax.set_yticks(range(len(order)), [atlas.names[i] for i in order], fontsize=6)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=10)
    present = [lobe for lobe in LOBE_ORDER if lobe in set(atlas.lobes)]
    ax.legend(handles=[Patch(color=LOBE_COLORS[lobe], label=lobe) for lobe in present], fontsize=6, loc="lower right")


# --- Graphe dans le cerveau -------------------------------------------------------------------------------
# Nœuds aux centroïdes de `atlas.coords_mm` (MNI pour les régions), surface fsaverage en verre.
# Un graphe orienté (nx.DiGraph) reçoit une flèche par arête, posée aux 2/3 du segment, vers la cible.

type NodeValues = Sequence[float] | NDArray[np.floating]
type NodeGroups = Sequence[int] | NDArray[np.integer]

_ARROW_AT = 0.65  # position de la flèche le long de l'arête (0 = source, 1 = cible)


def _edge_arrays(g: nx.Graph, atlas: Atlas) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Extrémités source et cible (n_edges, 3) et poids (n_edges,), triés par poids croissant."""
    pos = dict(zip(atlas.names, atlas.coords_mm, strict=True))
    edges = sorted(g.edges(data="weight", default=1.0), key=lambda e: e[2])
    src = np.array([pos[u] for u, _, _ in edges], dtype=np.float64).reshape(-1, 3)
    dst = np.array([pos[v] for _, v, _ in edges], dtype=np.float64).reshape(-1, 3)
    return src, dst, np.array([w for *_, w in edges], dtype=np.float64)


def _node_style(
    g: nx.Graph, atlas: Atlas, groups: NodeGroups | None, node_values: NodeValues | None
) -> tuple[list[str], list[str], NDArray[np.float64]]:
    """Couleur et légende par nœud (communauté si `groups`, sinon lobe), valeurs (défaut : strength)."""
    if groups is None:
        colors = [LOBE_COLORS[lobe] for lobe in atlas.lobes]
        legend = list(atlas.lobes)
    else:
        cmap = plt.get_cmap("tab10")  # mêmes couleurs que plot_circular_graph
        colors = [to_hex(cmap(int(c) % 10)) for c in groups]
        legend = [f"communauté {int(c)}" for c in groups]
    if node_values is None:
        values = np.array([g.degree(n, weight="weight") for n in atlas.names], dtype=np.float64)
    else:
        values = np.asarray(node_values, dtype=np.float64)
    return colors, legend, values


def _legend_order(legend: list[str]) -> list[str]:
    """Lobes dans l'ordre antéro-postérieur ; communautés par effectif décroissant."""
    counts = {k: legend.count(k) for k in dict.fromkeys(legend)}
    return sorted(counts, key=lambda k: (LOBE_ORDER.index(k) if k in LOBE_ORDER else len(LOBE_ORDER), -counts[k], k))


def plot_brain_graph_3d(
    g: nx.Graph,
    atlas: Atlas,
    mesh: BrainMesh | None = None,
    groups: NodeGroups | None = None,
    node_values: NodeValues | None = None,
    title: str = "",
    n_weight_bins: int = 4,
) -> go.Figure:
    """Graphe interactif (plotly) : rotation, zoom, survol des nœuds.

    Taille des nœuds = `node_values` (défaut : strength), couleur = `groups` (ex. communautés) ou lobe ;
    `node_values` et `groups` sont alignés sur `atlas.names`.
    Arêtes regroupées en `n_weight_bins` quantiles de poids : plotly n'accepte qu'une épaisseur par trace.
    """
    fig = go.Figure()
    if mesh is not None:
        v, t = mesh.vertices_mm.astype(np.float32), mesh.triangles.astype(np.int32)
        fig.add_trace(
            go.Mesh3d(
                x=v[:, 0],
                y=v[:, 1],
                z=v[:, 2],
                i=t[:, 0],
                j=t[:, 1],
                k=t[:, 2],
                color="#c8c8c8",
                opacity=0.12,
                hoverinfo="skip",
                showlegend=False,
                lighting={"ambient": 0.55, "diffuse": 0.7, "specular": 0.05},
            )
        )
    src, dst, w = _edge_arrays(g, atlas)
    if w.size:
        edges_q = np.quantile(w, np.linspace(0, 1, n_weight_bins + 1))
        level = np.clip(np.searchsorted(edges_q, w, side="right") - 1, 0, n_weight_bins - 1)
        for b in range(n_weight_bins):
            sel = level == b
            if not sel.any():
                continue
            nan = np.full_like(src[sel], np.nan)
            alpha = 0.2 + 0.6 * b / max(n_weight_bins - 1, 1)
            xyz = np.stack([src[sel], dst[sel], nan], axis=1).reshape(-1, 3)  # segments séparés par NaN
            fig.add_trace(
                go.Scatter3d(
                    x=xyz[:, 0],
                    y=xyz[:, 1],
                    z=xyz[:, 2],
                    mode="lines",
                    hoverinfo="skip",
                    line={"width": 1.5 + 2.5 * b, "color": f"rgba(40, 40, 40, {alpha:.2f})"},
                    name=f"poids {edges_q[b]:.2f}-{edges_q[b + 1]:.2f}",
                    legendgroup="arêtes",
                    legendgrouptitle_text="arêtes" if b == 0 else None,
                )
            )
        if g.is_directed():
            d = dst - src
            unit = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            at = src + _ARROW_AT * d
            fig.add_trace(
                go.Cone(
                    x=at[:, 0],
                    y=at[:, 1],
                    z=at[:, 2],
                    u=unit[:, 0],
                    v=unit[:, 1],
                    w=unit[:, 2],
                    sizemode="raw",
                    sizeref=7.0,
                    anchor="center",
                    colorscale=[[0, "#282828"], [1, "#282828"]],
                    showscale=False,
                    hoverinfo="skip",
                    name="sens",
                    showlegend=True,
                )
            )
    colors, legend, values = _node_style(g, atlas, groups, node_values)
    sizes = 5 + 17 * values / max(float(values.max()), 1e-12)
    xyz = atlas.coords_mm
    for key in _legend_order(legend):
        idx = [i for i, k in enumerate(legend) if k == key]
        fig.add_trace(
            go.Scatter3d(
                x=xyz[idx, 0],
                y=xyz[idx, 1],
                z=xyz[idx, 2],
                mode="markers",
                name=f"{key} ({len(idx)})",
                legendgroup="nœuds",
                marker={"size": sizes[idx], "color": colors[idx[0]], "line": {"width": 0.5, "color": "black"}},
                text=[f"{atlas.names[i]}<br>{atlas.lobes[i]}<br>valeur = {values[i]:.3g}" for i in idx],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    hidden = {"visible": False}
    fig.update_layout(
        title=title,
        scene={
            "xaxis": hidden,
            "yaxis": hidden,
            "zaxis": hidden,
            "aspectmode": "data",
            "camera": {"eye": {"x": -1.35, "y": -0.55, "z": 0.75}},  # vue gauche, postéro-supérieure
        },
        margin={"l": 0, "r": 0, "t": 40, "b": 0},
        legend={"itemsizing": "constant"},
        paper_bgcolor="white",
    )
    return fig


# (axe horizontal, axe vertical) projetés : avant à droite (sagittale), gauche à gauche (convention neurologique).
_VIEWS: tuple[tuple[str, int, int], ...] = (("sagittale", 1, 2), ("coronale", 0, 2), ("axiale", 0, 1))


def plot_brain_graph_views(
    g: nx.Graph,
    atlas: Atlas,
    mesh: BrainMesh | None = None,
    groups: NodeGroups | None = None,
    node_values: NodeValues | None = None,
    title: str = "",
) -> Figure:
    """Version statique (rapport) : trois projections orthogonales du graphe dans un cerveau en verre."""
    colors, legend, values = _node_style(g, atlas, groups, node_values)
    sizes = 8 + 140 * values / max(float(values.max()), 1e-12)
    src, dst, w = _edge_arrays(g, atlas)
    rel = (w - w.min()) / max(float(np.ptp(w)), 1e-12) if w.size else w
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
    for ax, (name, h, v) in zip(axes, _VIEWS, strict=True):
        if mesh is not None:
            tris = mesh.vertices_mm[:, [h, v]][mesh.triangles]  # (n_triangles, 3, 2)
            ax.add_collection(PolyCollection(list(tris), facecolors="0.45", edgecolors="none", alpha=0.03))
        if w.size:
            segs = np.stack([src[:, [h, v]], dst[:, [h, v]]], axis=1)
            rgba = np.zeros((w.size, 4))
            rgba[:, 3] = 0.15 + 0.7 * rel
            ax.add_collection(LineCollection(list(segs), colors=rgba, linewidths=0.3 + 2.2 * rel))
            if g.is_directed():
                d = (dst - src)[:, [h, v]]
                at = src[:, [h, v]] + _ARROW_AT * d
                unit = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
                ax.quiver(
                    at[:, 0],
                    at[:, 1],
                    unit[:, 0],
                    unit[:, 1],
                    angles="xy",
                    scale_units="xy",
                    scale=1 / 8,  # flèches de 8 mm
                    pivot="mid",
                    width=0.0025,
                    headwidth=7,
                    headlength=8,
                    headaxislength=7,
                    color=np.column_stack([np.zeros((w.size, 3)), 0.45 + 0.55 * rel]),
                )
        xy = atlas.coords_mm[:, [h, v]]
        ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=colors, edgecolors="black", linewidths=0.4, zorder=3)
        ax.set_title(name, fontsize=10)
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.axis("off")
    handles = [Patch(color=colors[legend.index(k)], label=f"{k} ({legend.count(k)})") for k in _legend_order(legend)]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 9), fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11)
    return fig
