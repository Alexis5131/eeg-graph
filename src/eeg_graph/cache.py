"""Cache des matrices de connectivité : un .npz par (enregistrement, découpage) + index.csv global."""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from eeg_graph.nodes import Atlas

logger = logging.getLogger(__name__)


class CacheValidationError(ValueError):
    """Matrice de connectivité invalide (asymétrique, diagonale non nulle, NaN, hors [0, 1])."""


def validate_connectivity(conn: NDArray[np.floating], atol: float = 1e-5) -> None:
    """V2 : symétrie, diagonale nulle, finitude, valeurs dans [0, 1] sur les deux derniers axes."""
    if not np.isfinite(conn).all():
        raise CacheValidationError("NaN ou inf dans la connectivité")
    if not np.allclose(conn, np.swapaxes(conn, -1, -2), atol=atol):
        raise CacheValidationError("matrices non symétriques")
    diag = np.diagonal(conn, axis1=-2, axis2=-1)
    if np.abs(diag).max(initial=0.0) > atol:
        raise CacheValidationError("diagonale non nulle")
    if conn.size and (conn.min() < -atol or conn.max() > 1 + atol):
        raise CacheValidationError(f"valeurs hors [0, 1] : [{conn.min():.3g}, {conn.max():.3g}]")


@dataclass(frozen=True)
class RecordingResult:
    dataset: str
    subject: int
    run: int
    condition: str
    node_space: str
    epoch_spec: str
    sfreq: float
    n_times: int  # échantillons par epoch
    node_names: tuple[str, ...]
    node_hemispheres: tuple[str, ...]
    node_lobes: tuple[str, ...]
    node_coords_mm: NDArray[np.float64] = field(repr=False)
    bands: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    conn_epochs: NDArray[np.float32] = field(repr=False, default_factory=lambda: np.zeros(0, np.float32))
    power_epochs: NDArray[np.float32] = field(repr=False, default_factory=lambda: np.zeros(0, np.float32))
    epoch_onsets: NDArray[np.int64] = field(repr=False, default_factory=lambda: np.zeros(0, np.int64))
    epoch_labels: NDArray[np.str_] = field(repr=False, default_factory=lambda: np.zeros(0, str))
    n_epochs_total: int = 0
    ica_excluded: tuple[int, ...] = ()
    config_hash: str = ""
    versions: dict[str, str] = field(default_factory=dict)
    seconds: float = float("nan")

    @property
    def n_epochs(self) -> int:
        return int(self.conn_epochs.shape[0])

    @property
    def conn(self) -> NDArray[np.float32]:
        """Matrice de l'enregistrement = moyenne des epochs : (n_metrics, n_bands, n, n)."""
        return self.conn_epochs.mean(axis=0)

    def atlas(self) -> Atlas:
        return Atlas(
            space="source" if self.node_space == "source" else "sensor",
            names=self.node_names,
            hemispheres=self.node_hemispheres,
            lobes=self.node_lobes,
            coords_mm=self.node_coords_mm,
        )

    def matrix(self, metric: str, band: str) -> NDArray[np.float32]:
        return self.conn[self.metrics.index(metric), self.bands.index(band)]

    def meta(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "subject": self.subject,
            "run": self.run,
            "condition": self.condition,
            "node_space": self.node_space,
            "epoch_spec": self.epoch_spec,
            "sfreq": self.sfreq,
            "n_times": self.n_times,
            "n_epochs": self.n_epochs,
            "n_epochs_total": self.n_epochs_total,
            "ica_excluded": list(self.ica_excluded),
            "config_hash": self.config_hash,
            "versions": self.versions,
            "seconds": self.seconds,
        }


def result_path(cache_dir: Path, dataset: str, node_space: str, subject: int, run: int, epoch_spec: str) -> Path:
    return cache_dir / dataset / node_space / f"sub-{subject:03d}_run-{run:02d}_{epoch_spec}.npz"


def save_result(result: RecordingResult, path: Path) -> None:
    validate_connectivity(result.conn_epochs)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        conn=result.conn.astype(np.float32),
        conn_epochs=result.conn_epochs.astype(np.float32),
        power_epochs=result.power_epochs.astype(np.float32),
        epoch_onsets=result.epoch_onsets.astype(np.int64),
        epoch_labels=result.epoch_labels.astype(str),
        node_names=np.array(result.node_names),
        node_hemispheres=np.array(result.node_hemispheres),
        node_lobes=np.array(result.node_lobes),
        node_coords_mm=result.node_coords_mm,
        bands=np.array(result.bands),
        metrics=np.array(result.metrics),
        meta=np.array(json.dumps(result.meta())),
    )


def load_result(path: Path) -> RecordingResult:
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["meta"]))
        return RecordingResult(
            dataset=meta["dataset"],
            subject=int(meta["subject"]),
            run=int(meta["run"]),
            condition=meta["condition"],
            node_space=meta["node_space"],
            epoch_spec=meta["epoch_spec"],
            sfreq=float(meta["sfreq"]),
            n_times=int(meta["n_times"]),
            node_names=tuple(f["node_names"].tolist()),
            node_hemispheres=tuple(f["node_hemispheres"].tolist()),
            node_lobes=tuple(f["node_lobes"].tolist()),
            node_coords_mm=f["node_coords_mm"],
            bands=tuple(f["bands"].tolist()),
            metrics=tuple(f["metrics"].tolist()),
            conn_epochs=f["conn_epochs"],
            power_epochs=f["power_epochs"],
            epoch_onsets=f["epoch_onsets"],
            epoch_labels=f["epoch_labels"],
            n_epochs_total=int(meta["n_epochs_total"]),
            ica_excluded=tuple(meta["ica_excluded"]),
            config_hash=meta["config_hash"],
            versions=meta["versions"],
            seconds=float(meta["seconds"]),
        )


_INDEX_KEY = ["dataset", "node_space", "subject", "run", "epoch_spec"]


@contextmanager
def file_lock(path: Path, timeout: float = 120.0, poll: float = 0.05) -> Iterator[None]:
    """Verrou exclusif inter-processus : création atomique de `path` (O_CREAT | O_EXCL), portable.

    Les workers joblib de `build_cache` mettent à jour le même index.csv ; sans verrou, deux
    lecture-modification-écriture simultanées perdent une ligne. Un verrou plus vieux que `timeout`
    (processus tué) est considéré comme abandonné et retiré.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > timeout:
                    path.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"Verrou {path} non obtenu en {timeout} s") from None
            time.sleep(poll)
    try:
        yield
    finally:
        os.close(fd)
        path.unlink(missing_ok=True)


def update_index(index_path: Path, row: dict[str, object]) -> None:
    """Ajoute ou remplace la ligne (dataset, node_space, subject, run, epoch_spec) de l'index.

    Sûr en écriture concurrente (verrou + remplacement atomique du fichier).
    """
    with file_lock(index_path.with_suffix(".lock")):
        new = pd.DataFrame([row])
        if index_path.exists():
            old = pd.read_csv(index_path)
            key = tuple(row[k] for k in _INDEX_KEY)
            same = np.ones(len(old), dtype=bool)
            for k, v in zip(_INDEX_KEY, key, strict=True):
                same &= old[k].astype(str).to_numpy() == str(v)
            new = pd.concat([old.loc[~same], new], ignore_index=True)
        tmp = index_path.with_suffix(f".{os.getpid()}.tmp")
        new.sort_values(_INDEX_KEY).to_csv(tmp, index=False)
        tmp.replace(index_path)
