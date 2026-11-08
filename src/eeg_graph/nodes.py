"""Définition des nœuds du graphe : régions Desikan-Killiany (espace source) ou électrodes (capteurs)."""

from __future__ import annotations

import re
from dataclasses import dataclass

import mne
import numpy as np
from numpy.typing import NDArray

from eeg_graph.config import NodeSpace

LOBE_ORDER: tuple[str, ...] = (
    "frontal",
    "central",
    "cingulate",
    "insula",
    "temporal",
    "parietal",
    "occipital",
)

# Regroupement en lobes des 34 régions Desikan-Killiany (Desikan et al. 2006).
DK_LOBES: dict[str, str] = {
    **dict.fromkeys(
        (
            "caudalmiddlefrontal",
            "frontalpole",
            "lateralorbitofrontal",
            "medialorbitofrontal",
            "paracentral",
            "parsopercularis",
            "parsorbitalis",
            "parstriangularis",
            "precentral",
            "rostralmiddlefrontal",
            "superiorfrontal",
        ),
        "frontal",
    ),
    **dict.fromkeys(("inferiorparietal", "postcentral", "precuneus", "superiorparietal", "supramarginal"), "parietal"),
    **dict.fromkeys(
        (
            "bankssts",
            "entorhinal",
            "fusiform",
            "inferiortemporal",
            "middletemporal",
            "parahippocampal",
            "superiortemporal",
            "temporalpole",
            "transversetemporal",
        ),
        "temporal",
    ),
    **dict.fromkeys(("cuneus", "lateraloccipital", "lingual", "pericalcarine"), "occipital"),
    **dict.fromkeys(
        (
            "caudalanteriorcingulate",
            "isthmuscingulate",
            "posteriorcingulate",
            "rostralanteriorcingulate",
        ),
        "cingulate",
    ),
    "insula": "insula",
}

# Préfixes 10-10 -> lobe (le plus long préfixe gagne).
_SENSOR_PREFIX_LOBES: tuple[tuple[str, str], ...] = (
    ("Fp", "frontal"),
    ("AF", "frontal"),
    ("FT", "temporal"),
    ("FC", "central"),
    ("TP", "temporal"),
    ("CP", "parietal"),
    ("PO", "occipital"),
    ("F", "frontal"),
    ("C", "central"),
    ("T", "temporal"),
    ("P", "parietal"),
    ("O", "occipital"),
    ("I", "occipital"),
)


@dataclass(frozen=True)
class Atlas:
    """Nœuds du graphe et leurs attributs (hémisphère, lobe, coordonnées en mm)."""

    space: NodeSpace
    names: tuple[str, ...]
    hemispheres: tuple[str, ...]  # "lh" | "rh" | "mid"
    lobes: tuple[str, ...]
    coords_mm: NDArray[np.float64]  # (n_nodes, 3) : MNI pour les régions, repère tête pour les électrodes

    def __post_init__(self) -> None:
        n = len(self.names)
        if not (len(self.hemispheres) == len(self.lobes) == self.coords_mm.shape[0] == n):
            raise ValueError("Attributs de nœuds de longueurs différentes")

    @property
    def n_nodes(self) -> int:
        return len(self.names)

    def node_attrs(self) -> dict[str, list[object]]:
        return {
            "hemisphere": list(self.hemispheres),
            "lobe": list(self.lobes),
            "x": self.coords_mm[:, 0].tolist(),
            "y": self.coords_mm[:, 1].tolist(),
            "z": self.coords_mm[:, 2].tolist(),
        }

    def display_order(self) -> NDArray[np.int64]:
        """Ordre de tracé : hémisphère gauche, milieu, droit ; puis lobe (antéro-postérieur) ; puis y."""
        hemi_rank = {"lh": 0, "mid": 1, "rh": 2}
        keys = [
            (hemi_rank[h], LOBE_ORDER.index(lobe), -self.coords_mm[i, 1])
            for i, (h, lobe) in enumerate(zip(self.hemispheres, self.lobes, strict=True))
        ]
        return np.array(sorted(range(self.n_nodes), key=lambda i: keys[i]), dtype=np.int64)


def dk_region(label_name: str) -> tuple[str, str]:
    """'precuneus-lh' -> ('precuneus', 'lh')."""
    region, hemi = label_name.rsplit("-", 1)
    return region, hemi


def sensor_lobe(ch_name: str) -> str:
    for prefix, lobe in _SENSOR_PREFIX_LOBES:
        if ch_name.startswith(prefix):
            return lobe
    raise ValueError(f"Nom d'électrode hors 10-10 : {ch_name}")


def sensor_hemisphere(ch_name: str) -> str:
    match = re.search(r"(\d+|z)$", ch_name)
    if match is None:
        raise ValueError(f"Nom d'électrode hors 10-10 : {ch_name}")
    suffix = match.group(1)
    if suffix == "z":
        return "mid"
    return "lh" if int(suffix) % 2 == 1 else "rh"


def sensor_atlas(info: mne.Info) -> Atlas:
    """Nœuds = électrodes EEG (montage requis pour les coordonnées)."""
    picks = mne.pick_types(info, eeg=True, exclude="bads")
    names = tuple(info["ch_names"][i] for i in picks)
    coords = np.array([info["chs"][i]["loc"][:3] for i in picks], dtype=np.float64) * 1e3
    if not np.isfinite(coords).all() or np.allclose(coords, 0):
        raise ValueError("Positions d'électrodes absentes : appliquer un montage avant sensor_atlas")
    return Atlas(
        space="sensor",
        names=names,
        hemispheres=tuple(sensor_hemisphere(n) for n in names),
        lobes=tuple(sensor_lobe(n) for n in names),
        coords_mm=coords,
    )
