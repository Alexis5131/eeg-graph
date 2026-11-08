"""Espace source sur le template fsaverage : forward, inverse, et noyau régions × électrodes.

Modèle direct : tutoriel MNE « EEG forward operator with a template MRI »
(https://mne.tools/stable/auto_tutorials/forward/35_eeg_no_mri.html).

Noyau W (n_labels × n_channels) : avec un template commun, une covariance ad hoc et le même jeu
d'électrodes, l'opérateur inverse est identique pour tous les enregistrements. L'inverse
(`pick_ori="normal"`) et l'extraction `mean_flip` sont linéaires, donc
  label_ts = W @ X,  avec W = extract_label_time_course(apply_inverse(I)).
C'est la technique qu'utilise MNE dans `make_inverse_resolution_matrix` (inverse appliqué à une
`EvokedArray` identité). Le projecteur de référence moyenne est inclus dans le noyau inverse ;
`nave=1` comme dans `apply_inverse_epochs`. Équivalence vérifiée par tests/test_source_kernel.py.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
from mne.minimum_norm import apply_inverse, make_inverse_operator
from numpy.typing import NDArray

from eeg_graph.config import InverseConfig, Paths
from eeg_graph.nodes import DK_LOBES, Atlas, dk_region

logger = logging.getLogger(__name__)

N_DK_LABELS = 68


class SourceModelError(RuntimeError):
    """Incohérence entre les données et le modèle source (canaux, atlas)."""


def fsaverage_dir(paths: Paths = Paths()) -> Path:
    paths.subjects_dir.mkdir(parents=True, exist_ok=True)
    return Path(mne.datasets.fetch_fsaverage(subjects_dir=paths.subjects_dir, verbose="error"))


def eeg_channel_names(info: mne.Info) -> list[str]:
    return [info["ch_names"][i] for i in mne.pick_types(info, eeg=True, exclude="bads")]


def _key(*parts: object) -> str:
    return hashlib.sha1(repr(parts).encode()).hexdigest()[:12]


def load_dk_labels(paths: Paths = Paths(), parc: str = "aparc") -> list[mne.Label]:
    """68 régions Desikan-Killiany : hémisphère gauche puis droit, ordre alphabétique."""
    fsaverage_dir(paths)
    labels = mne.read_labels_from_annot("fsaverage", parc=parc, subjects_dir=paths.subjects_dir, verbose="error")
    # `unknown-lh` existe sur fsaverage et casse extract_label_time_course.
    labels = [lab for lab in labels if not lab.name.startswith("unknown")]
    labels.sort(key=lambda lab: (lab.hemi, lab.name))
    if len(labels) != N_DK_LABELS:
        raise SourceModelError(f"{len(labels)} labels au lieu de {N_DK_LABELS}")
    return labels


def make_or_load_forward(info: mne.Info, cfg: InverseConfig, paths: Paths = Paths()) -> mne.Forward:
    """Forward fsaverage (BEM 3 couches 5120, source ico-5), mis en cache par jeu de canaux."""
    fs_dir = fsaverage_dir(paths)
    ch_names = eeg_channel_names(info)
    fname = paths.derivatives / f"fsaverage-{_key(ch_names, cfg.montage, cfg.mindist)}-fwd.fif"
    if fname.exists():
        return mne.read_forward_solution(fname, verbose="error")
    logger.info("Calcul du forward fsaverage (%d électrodes)", len(ch_names))
    fwd = mne.make_forward_solution(
        mne.pick_info(info, mne.pick_channels(info["ch_names"], ch_names)),
        trans="fsaverage",
        src=fs_dir / "bem" / "fsaverage-ico-5-src.fif",
        bem=fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif",
        eeg=True,
        meg=False,
        mindist=cfg.mindist,
        verbose="error",
    )
    fname.parent.mkdir(parents=True, exist_ok=True)
    mne.write_forward_solution(fname, fwd, overwrite=True, verbose="error")
    return fwd


def make_inverse(info: mne.Info, fwd: mne.Forward, cfg: InverseConfig) -> mne.minimum_norm.InverseOperator:
    cov = mne.make_ad_hoc_cov(info, verbose="error")
    return make_inverse_operator(info, fwd, cov, loose=cfg.loose, depth=cfg.depth, verbose="error")


def source_atlas(labels: list[mne.Label], paths: Paths = Paths()) -> Atlas:
    """Attributs des régions : hémisphère, lobe, centroïde en coordonnées MNI (mm)."""
    coords = np.empty((len(labels), 3))
    for i, lab in enumerate(labels):
        vertex = lab.center_of_mass(subject="fsaverage", subjects_dir=paths.subjects_dir)
        hemi = 0 if lab.hemi == "lh" else 1
        coords[i] = mne.vertex_to_mni(vertex, hemi, "fsaverage", subjects_dir=paths.subjects_dir)
    regions = [dk_region(lab.name) for lab in labels]
    return Atlas(
        space="source",
        names=tuple(lab.name for lab in labels),
        hemispheres=tuple(h for _, h in regions),
        lobes=tuple(DK_LOBES[r] for r, _ in regions),
        coords_mm=coords,
    )


@dataclass(frozen=True)
class BrainMesh:
    """Surface corticale triangulée des deux hémisphères, en coordonnées MNI (mm)."""

    vertices_mm: NDArray[np.float64]  # (n_vertices, 3)
    triangles: NDArray[np.int64]  # (n_triangles, 3), indices dans vertices_mm


def fsaverage_mesh(paths: Paths = Paths(), surface: str = "pial", spacing: str = "ico5") -> BrainMesh:
    """Surface fsaverage décimée (ico-5 : 10 242 sommets par hémisphère) pour les figures 3D.

    Sur fsaverage, `vertex_to_mni` renvoie les coordonnées surface-RAS telles quelles : la surface est dans
    le repère des centroïdes de `source_atlas`, sans transformation.
    """
    fsaverage_dir(paths)
    src = mne.setup_source_space(
        "fsaverage", spacing=spacing, surface=surface, add_dist=False, subjects_dir=paths.subjects_dir, verbose="error"
    )
    vertices, triangles, offset = [], [], 0
    for hemi in src:
        used = hemi["vertno"]
        remap = np.full(hemi["np"], -1, dtype=np.int64)
        remap[used] = np.arange(used.size)
        vertices.append(hemi["rr"][used] * 1e3)
        triangles.append(remap[hemi["use_tris"]] + offset)
        offset += used.size
    return BrainMesh(np.vstack(vertices), np.vstack(triangles))


@dataclass(frozen=True)
class LabelKernel:
    weights: NDArray[np.float64]  # (n_labels, n_channels)
    ch_names: tuple[str, ...]
    atlas: Atlas
    method: str
    lambda2: float

    def project(self, raw: mne.io.BaseRaw) -> NDArray[np.float64]:
        """Séries continues des régions (n_labels, n_times)."""
        names = eeg_channel_names(raw.info)
        if tuple(names) != self.ch_names:
            raise SourceModelError("Canaux EEG différents de ceux du noyau : recalculer W")
        return self.weights @ raw.get_data(picks=list(self.ch_names))


def compute_label_kernel(
    info: mne.Info,
    inverse_operator: mne.minimum_norm.InverseOperator,
    labels: list[mne.Label],
    cfg: InverseConfig,
) -> NDArray[np.float64]:
    """W = mean_flip(inverse(I)) : réponse des régions à chaque électrode unitaire."""
    ch_names = eeg_channel_names(info)
    eeg_info = mne.pick_info(info, mne.pick_channels(info["ch_names"], ch_names))
    identity = mne.EvokedArray(np.eye(len(ch_names)), eeg_info, tmin=0.0, nave=1, verbose="error")
    stc = apply_inverse(
        identity, inverse_operator, lambda2=cfg.lambda2, method=cfg.method, pick_ori="normal", verbose="error"
    )
    weights = mne.extract_label_time_course(stc, labels, inverse_operator["src"], mode=cfg.label_mode, verbose="error")
    return np.asarray(weights, dtype=np.float64)


def make_label_kernel(info: mne.Info, cfg: InverseConfig, paths: Paths = Paths()) -> LabelKernel:
    """Noyau W mis en cache (clé : canaux, montage, méthode, λ², loose, depth, parcellisation)."""
    ch_names = tuple(eeg_channel_names(info))
    labels = load_dk_labels(paths, cfg.parc)
    key = _key(ch_names, cfg)
    fname = paths.derivatives / f"label-kernel-{key}.npz"
    if fname.exists():
        weights = np.load(fname)["weights"]
    else:
        logger.info("Calcul du noyau %s (λ²=%g) sur %d régions", cfg.method, cfg.lambda2, len(labels))
        fwd = make_or_load_forward(info, cfg, paths)
        inv = make_inverse(info, fwd, cfg)
        weights = compute_label_kernel(info, inv, labels, cfg)
        fname.parent.mkdir(parents=True, exist_ok=True)
        np.savez(fname, weights=weights, ch_names=np.array(ch_names))
    return LabelKernel(weights, ch_names, source_atlas(labels, paths), cfg.method, cfg.lambda2)
