from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from eeg_graph.cache import RecordingResult
from eeg_graph.config import Paths
from eeg_graph.task_analysis import (
    CONDITIONS,
    cohort_table,
    condition_epochs,
    equalize_and_split,
    load_cohort_matrices,
    save_cohort_matrices,
    subject_matrices,
    subject_rng,
)

N = 3  # nœuds


def _result(run: int, condition: str, labels: list[str], subject: int = 1, offset: float = 0.0) -> RecordingResult:
    """conn_epochs[e] vaut (offset + e) / 1000 hors diagonale : chaque epoch est identifiable par sa valeur."""
    n_epochs = len(labels)
    values = (offset + np.arange(n_epochs)) / 1000
    conn = np.broadcast_to(values[:, None, None, None, None], (n_epochs, 2, 2, N, N)).copy()
    conn[..., np.arange(N), np.arange(N)] = 0.0
    conn[:, 1] *= 0.5  # 2e métrique distincte
    return RecordingResult(
        dataset="test",
        subject=subject,
        run=run,
        condition=condition,
        node_space="source",
        epoch_spec="task_lr",
        sfreq=160.0,
        n_times=560,
        node_names=("a-lh", "b-lh", "c-rh"),
        node_hemispheres=("lh", "lh", "rh"),
        node_lobes=("frontal", "parietal", "occipital"),
        node_coords_mm=np.zeros((N, 3)),
        bands=("alpha", "beta"),
        metrics=("wpli", "ciplv"),
        conn_epochs=conn.astype(np.float32),
        power_epochs=np.zeros((n_epochs, 2, N), dtype=np.float32),
        epoch_onsets=np.arange(n_epochs, dtype=np.int64),
        epoch_labels=np.array(labels),
        n_epochs_total=n_epochs,
    )


def test_equalize_and_split_sizes_and_disjoint_halves() -> None:
    counts = {"MM": 30, "MI": 25, "REST": 80}
    sel = equalize_and_split(counts, subject_rng(1))
    assert set(sel) == set(counts)
    for cond, s in sel.items():
        assert len(s.full) == 25 and len(np.unique(s.full)) == 25
        assert s.full.max() < counts[cond]
        assert len(s.half_a) == len(s.half_b) == 12
        assert not set(s.half_a) & set(s.half_b)
        assert set(s.half_a) | set(s.half_b) <= set(s.full)


def test_equalize_and_split_reproducible_and_order_independent() -> None:
    a = equalize_and_split({"MM": 30, "MI": 25, "REST": 80}, subject_rng(7))
    b = equalize_and_split({"REST": 80, "MI": 25, "MM": 30}, subject_rng(7))
    c = equalize_and_split({"MM": 30, "MI": 25, "REST": 80}, subject_rng(8))
    for cond in CONDITIONS:
        np.testing.assert_array_equal(a[cond].full, b[cond].full)
        np.testing.assert_array_equal(a[cond].half_a, b[cond].half_a)
    assert any(not np.array_equal(a[cond].full, c[cond].full) for cond in CONDITIONS)


def test_equalize_and_split_odd_and_too_small() -> None:
    sel = equalize_and_split({"MM": 5, "MI": 9, "REST": 9}, subject_rng(1))
    assert len(sel["MM"].full) == 5 and len(sel["MM"].half_a) == 2
    np.testing.assert_array_equal(sel["MM"].full, np.arange(5))  # tout est pris dans la plus petite condition
    with pytest.raises(ValueError):
        equalize_and_split({"MM": 1, "MI": 9, "REST": 9}, subject_rng(1))


def test_condition_epochs_grouping() -> None:
    mm = _result(3, "mm_lr", ["rest", "left", "right", "rest"])
    mi = _result(4, "mi_lr", ["left", "rest", "right"], offset=100)
    ep = condition_epochs([mm, mi], metric="wpli", bands=("beta",))
    assert ep.counts() == {"MM": 2, "MI": 2, "REST": 3}
    np.testing.assert_allclose(ep.conn["MM"][:, 0, 0, 1], [0.001, 0.002])
    np.testing.assert_allclose(ep.conn["MI"][:, 0, 0, 1], [0.100, 0.102])
    np.testing.assert_allclose(ep.conn["REST"][:, 0, 0, 1], [0.000, 0.003, 0.101])
    assert ep.runs["REST"].tolist() == [3, 3, 4]
    assert ep.conn["MM"].shape == (2, 1, N, N)


def test_subject_matrices_means_match_selection() -> None:
    mm = _result(3, "mm_lr", ["left"] * 6 + ["rest"] * 6)
    mi = _result(4, "mi_lr", ["right"] * 9 + ["rest"] * 6, offset=100)
    ep = condition_epochs([mm, mi])
    out = subject_matrices(ep, seed=42)
    sel = equalize_and_split(ep.counts(), subject_rng(1, 42))
    assert out.n_epochs == 6 and out.n_half == 3
    assert out.matrices.shape == (3, 2, N, N) and out.halves.shape == (3, 2, 2, N, N)
    for k, cond in enumerate(CONDITIONS):
        np.testing.assert_allclose(out.matrices[k], ep.conn[cond][sel[cond].full].mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(out.halves[k, 0], ep.conn[cond][sel[cond].half_a].mean(axis=0), rtol=1e-6)
        np.testing.assert_allclose(out.halves[k, 1], ep.conn[cond][sel[cond].half_b].mean(axis=0), rtol=1e-6)
    assert np.all(out.matrices[:, :, np.arange(N), np.arange(N)] == 0)


def test_save_and_load_cohort_matrices(tmp_path: Path) -> None:
    items = []
    for s in (1, 2):
        mm = _result(3, "mm_lr", ["left"] * 4 + ["rest"] * 4, subject=s)
        mi = _result(4, "mi_lr", ["right"] * 5 + ["rest"] * 4, subject=s, offset=50)
        ep = condition_epochs([mm, mi])
        items.append(subject_matrices(ep))
    path = tmp_path / "m.npz"
    meta = save_cohort_matrices(path, items, ep.atlas, "wpli", ("alpha", "beta"))
    assert meta.n_equalized.tolist() == [4, 4]
    assert pd.read_csv(path.with_suffix(".csv")).n_MI_available.tolist() == [5, 5]
    loaded = load_cohort_matrices(path)
    assert loaded.matrices.shape == (2, 3, 2, N, N) and loaded.halves.shape == (2, 3, 2, 2, N, N)
    np.testing.assert_array_equal(loaded.matrix(2, "MI", "beta"), items[1].matrices[1, 1])
    assert loaded.atlas.names == ep.atlas.names


def test_cohort_table_inclusion(tmp_path: Path) -> None:
    paths = Paths(root=tmp_path)
    rows = []

    def add(subject: int, run: int, status: str, labels: list[str] | None = None, reason: str = "") -> None:
        rel = f"c/s{subject}_r{run}.npz"
        if labels is not None:
            (tmp_path / "c").mkdir(exist_ok=True)
            np.savez(tmp_path / rel, epoch_labels=np.array(labels))
        rows.append(
            {"dataset": "eegmmidb", "node_space": "source", "subject": subject, "run": run, "epoch_spec": "task_lr"}
            | {"status": status, "reason": reason, "path": rel if labels is not None else np.nan}
        )

    ok = ["left"] * 8 + ["right"] * 7 + ["rest"] * 15
    for run in (3, 4, 7, 8, 11, 12):
        add(1, run, "ok", ok)
    for run in (3, 7, 11, 4):
        add(2, run, "ok", ok)
    for run in (8, 12):
        add(2, run, "excluded", reason="sfreq 128 Hz != 160 Hz")
    for run in (3, 4, 7, 8):
        add(3, run, "ok", ["left"] * 9 + ["rest"] * 15)  # 18 epochs tâche par condition
    table = cohort_table(pd.DataFrame(rows), [1, 2, 3, 4], paths).set_index("subject")
    assert table.included.tolist() == [True, False, False, False]
    assert table.loc[1, "MM_task_epochs"] == 45 and table.loc[1, "REST_epochs"] == 90
    assert "MI : 1 run(s) exploitable(s) < 2" in str(table.loc[2, "exclusion_reason"])
    assert "R08 excluded: sfreq 128 Hz" in str(table.loc[2, "runs_not_ok"])
    assert table.loc[3, "exclusion_reason"] == "MM : 18 epochs tâche < 20; MI : 18 epochs tâche < 20"
    assert table.loc[4, "exclusion_reason"] == "absent de index.csv"
