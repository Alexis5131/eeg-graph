from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eeg_graph.cache import (
    CacheValidationError,
    RecordingResult,
    load_result,
    save_result,
    update_index,
    validate_connectivity,
)


def _result(conn_epochs: np.ndarray) -> RecordingResult:
    n_epochs, _, _, n, _ = conn_epochs.shape
    return RecordingResult(
        dataset="test",
        subject=1,
        run=2,
        condition="rest_ec",
        node_space="sensor",
        epoch_spec="rest4s",
        sfreq=160.0,
        n_times=640,
        node_names=tuple(f"N{i}" for i in range(n)),
        node_hemispheres=("lh",) * n,
        node_lobes=("frontal",) * n,
        node_coords_mm=np.zeros((n, 3)),
        bands=("alpha", "beta"),
        metrics=("wpli",),
        conn_epochs=conn_epochs,
        power_epochs=np.zeros((n_epochs, 2, n), dtype=np.float32),
        epoch_onsets=np.arange(n_epochs, dtype=np.int64) * 640,
        epoch_labels=np.array(["rest"] * n_epochs),
        n_epochs_total=n_epochs + 1,
        ica_excluded=(0, 3),
        config_hash="abc",
        versions={"mne": "x"},
        seconds=1.5,
    )


def _valid_conn(n_epochs: int = 3, n: int = 4) -> np.ndarray:
    rng = np.random.default_rng(0)
    a = rng.random((n_epochs, 1, 2, n, n)).astype(np.float32)
    a = np.triu(a, 1)
    return a + np.swapaxes(a, -1, -2)


def test_round_trip(tmp_path: Path) -> None:
    original = _result(_valid_conn())
    path = tmp_path / "x.npz"
    save_result(original, path)
    loaded = load_result(path)
    np.testing.assert_array_equal(loaded.conn_epochs, original.conn_epochs)
    np.testing.assert_allclose(loaded.conn, original.conn_epochs.mean(axis=0))
    assert loaded.ica_excluded == (0, 3)
    assert loaded.epoch_labels.tolist() == ["rest"] * 3
    assert loaded.atlas().names == original.node_names
    assert loaded.matrix("wpli", "beta").shape == (4, 4)


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda c: c.__setitem__((0, 0, 0, 0, 1), 0.9),  # asymétrie
        lambda c: c.__setitem__((0, 0, 0, 2, 2), 0.5),  # diagonale
        lambda c: c.__setitem__((0, 0, 0, 1, 0), np.nan),
    ],
)
def test_validation_rejects_corrupted(corrupt: Callable[[np.ndarray], None]) -> None:
    conn = _valid_conn()
    corrupt(conn)
    with pytest.raises(CacheValidationError):
        validate_connectivity(conn)


def test_validation_rejects_out_of_range() -> None:
    with pytest.raises(CacheValidationError):
        validate_connectivity(_valid_conn() * 3)


def test_update_index_replaces_same_key(tmp_path: Path) -> None:
    index = tmp_path / "index.csv"
    base = {"dataset": "d", "node_space": "source", "subject": 1, "run": 1, "epoch_spec": "rest4s"}
    update_index(index, base | {"status": "error"})
    update_index(index, base | {"status": "ok"})
    update_index(index, base | {"run": 2, "status": "ok"})
    df = pd.read_csv(index)
    assert len(df) == 2
    assert df.loc[df.run == 1, "status"].item() == "ok"


def _write_rows(index: Path, subject: int, n: int) -> None:
    for run in range(n):
        update_index(index, {"dataset": "d", "node_space": "s", "subject": subject, "run": run, "epoch_spec": "e"})


def test_update_index_concurrent_writers(tmp_path: Path) -> None:
    """Plusieurs processus écrivent le même index : aucune ligne perdue."""
    from concurrent.futures import ProcessPoolExecutor

    index = tmp_path / "index.csv"
    with ProcessPoolExecutor(4) as pool:
        list(pool.map(_write_rows, [index] * 4, range(4), [15] * 4))
    assert len(pd.read_csv(index)) == 60
    assert not list(tmp_path.glob("*.lock")) and not list(tmp_path.glob("*.tmp"))
