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
        power_epochs=np.broadcast_to(values[:, None, None], (n_epochs, 2, N)).astype(np.float32),
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
    assert out.power is not None and out.power.shape == (3, 2, N)
    for k, cond in enumerate(CONDITIONS):
        assert ep.power is not None
        np.testing.assert_allclose(out.power[k], ep.power[cond][sel[cond].full].mean(axis=0), rtol=1e-6)


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
    assert loaded.power.shape == (2, 3, 2, N)


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


# --- Analyse ----------------------------------------------------------------------------------------------

from eeg_graph.nodes import Atlas  # noqa: E402
from eeg_graph.task_analysis import (  # noqa: E402
    bootstrap_median_ci,
    describe_graph,
    edge_spearman,
    global_metric_tests,
    graph_global_metrics,
    holm,
    jaccard,
    power_strength_confound,
    rank_biserial,
    robustness_table,
    sensorimotor_module,
    split_half_similarity,
    strength_spearman,
    top_hubs,
    top_regions,
)
from eeg_graph.validation import MOTOR_ROI  # noqa: E402

_NAMES = (*MOTOR_ROI, "x1-lh", "x2-lh", "x3-rh", "x4-rh")
_ATLAS = Atlas(
    space="source",
    names=_NAMES,
    hemispheres=tuple(n.rsplit("-", 1)[1] for n in _NAMES),
    lobes=("frontal",) * 2 + ("parietal",) * 2 + ("frontal",) * 2 + ("temporal",) * 2 + ("occipital",) * 2,
    coords_mm=np.zeros((len(_NAMES), 3)),
)


def _sym(rng: np.random.Generator, n: int) -> np.ndarray:
    a = np.triu(rng.random((n, n)), 1)
    return a + a.T


def test_holm_and_rank_biserial() -> None:
    np.testing.assert_allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    np.testing.assert_allclose(holm([0.5, 0.9]), [1.0, 1.0])
    assert rank_biserial(np.array([3.0, 0.0, 6.0]), np.zeros(3)) == pytest.approx(1.0)
    assert rank_biserial(np.array([1.0, -2.0, 3.0]), np.zeros(3)) == pytest.approx(1 / 3)
    assert rank_biserial(np.array([1.0, 2.0]), np.array([1.0, 2.0])) == 0.0


def test_bootstrap_median_ci() -> None:
    assert bootstrap_median_ci(np.full(20, 0.7)) == pytest.approx((0.7, 0.7, 0.7))
    x = np.random.default_rng(0).normal(size=50)
    med, lo, hi = bootstrap_median_ci(x)
    assert lo <= med <= hi
    assert bootstrap_median_ci(x) == (med, lo, hi)  # reproductible (SEED)


def test_split_half_similarity() -> None:
    rng = np.random.default_rng(0)
    base, other = _sym(rng, 10), _sym(rng, 10)
    noise = [0.05 * _sym(rng, 10) for _ in range(4)]
    halves = np.stack([[base + noise[0], base + noise[1]], [other + noise[2], other + noise[3]]])  # (2 cond, 2, n, n)
    out = split_half_similarity(halves, ("MM", "MI"), "MM", "MI", edge_spearman)
    assert out["within"] > 0.9 and abs(out["between"]) < 0.5
    assert out["ratio"] == pytest.approx(out["between"] / out["within"])
    same = split_half_similarity(np.stack([halves[0], halves[0]]), ("MM", "MI"), "MM", "MI", strength_spearman(0.3))
    assert same["ratio"] == pytest.approx(same["between"] / same["within"])


def test_describe_graph_two_blocks() -> None:
    w = 0.001 * _sym(np.random.default_rng(4), 10)  # fond faible : graphe complet connexe
    w[:4, :4] = 0.9
    w[4:8, 4:8] = 0.8
    np.fill_diagonal(w, 0)
    desc = describe_graph(w, _ATLAS, density=12 / 45, n_louvain_runs=5)  # garde les 12 arêtes des deux blocs
    rec = desc.record
    assert rec["n_edges"] == 12 and rec["n_components"] == 4
    assert rec["isolated_nodes"] == "x3-rh, x4-rh"
    assert rec["diameter_largest_component"] == 1
    assert rec["n_edges_mst_union"] == 15  # + 3 arêtes de l'arbre pour relier 4 composantes
    assert int(str(rec["diameter_mst_union"])) >= 2
    assert int(str(rec["n_communities"])) >= 2
    tops = top_regions(desc.metrics, k=4)
    assert set(tops.strength) == set(_NAMES[:4])


def test_graph_global_metrics_ranges() -> None:
    out = graph_global_metrics(_sym(np.random.default_rng(1), 12), 0.3, n_louvain_runs=3)
    assert set(out) == {"global_efficiency_weighted", "clustering_weighted", "modularity"}
    assert 0 < out["global_efficiency_weighted"] < 1 and 0 <= out["clustering_weighted"] <= 1


def test_hubs_modules_and_confound() -> None:
    strength = np.arange(10.0)
    assert top_hubs(strength, _ATLAS, k=2) == ["x4-rh", "x3-rh"]
    assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
    labels = np.array([0, 0, 0, 1, 1, 1, 0, 1, 2, 2])  # ROI : 3 dans 0, 3 dans 1 ; ex aequo -> plus petite
    assert sensorimotor_module(labels, _ATLAS) == ["precentral-lh", "precentral-rh", "postcentral-lh", "x1-lh"]
    labels = np.array([0, 0, 0, 0, 1, 1, 0, 1, 2, 2])
    assert sensorimotor_module(labels, _ATLAS) == [
        "precentral-lh",
        "precentral-rh",
        "postcentral-lh",
        "postcentral-rh",
        "x1-lh",
    ]
    rng = np.random.default_rng(2)
    power = rng.normal(size=(15, 3, 10))
    strength_s = np.repeat(power[:, :, :], 1, axis=0) * 2 + 1  # Δ strength monotone en Δ puissance
    out = power_strength_confound(power, strength_s, _ATLAS)
    assert out["rho"] == pytest.approx(1.0) and out["n"] == 15


def test_global_metric_tests_and_robustness() -> None:
    rng = np.random.default_rng(3)
    rows = []
    for s in range(12):
        for d in (0.1, 0.15):
            for cond, shift in (("MM", 0.2), ("MI", 0.1), ("REST", 0.0)):
                rows.append(
                    {"subject": s, "condition": cond, "band": "alpha", "density": d}
                    | {
                        m: shift + 0.01 * rng.normal()
                        for m in ("global_efficiency_weighted", "clustering_weighted", "modularity")
                    }
                )
    tests = global_metric_tests(pd.DataFrame(rows))
    assert len(tests) == 18  # 2 densités × 3 métriques × 3 comparaisons
    assert (tests.r_rb == 1).all() and (tests.p_holm >= tests.p).all()
    sims = pd.DataFrame(
        [
            {
                "band": "alpha",
                "density": np.nan,
                "measure": "edges",
                "comparison": "MM vs MI",
                "r_rb": -1.0,
                "p_holm": 0.001,
            },
            {
                "band": "alpha",
                "density": 0.1,
                "measure": "strength",
                "comparison": "MM vs MI",
                "r_rb": 0.2,
                "p_holm": 0.5,
            },
            {
                "band": "alpha",
                "density": 0.15,
                "measure": "strength",
                "comparison": "MM vs MI",
                "r_rb": -0.3,
                "p_holm": 0.2,
            },
        ]
    )
    table = robustness_table(tests, sims)
    assert list(table.columns) == ["alpha 10 %", "alpha 15 %"]
    assert table.loc["similarité edges : MM vs MI (entre vs intra)"].str.endswith("✓").all()
    assert str(table.loc["similarité strength : MM vs MI (entre vs intra)", "alpha 10 %"]).endswith("✗")
    assert str(table.loc["modularity : MM vs MI", "alpha 10 %"]).startswith("+")
