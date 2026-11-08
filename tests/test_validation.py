from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from eeg_graph.cache import RecordingResult, result_path, save_result
from eeg_graph.config import Paths
from eeg_graph.nodes import Atlas
from eeg_graph.validation import (
    MOTOR_ROI,
    cohort_erd,
    epoch_label_counts,
    erd,
    erd_by_subject,
    mm_vs_mi_test,
    roi_erd_negative_test,
    roi_mean,
    roi_top_check,
    task_erd_report,
)

_NAMES = (*MOTOR_ROI, *(f"r{i}-lh" for i in range(14)))  # 6 régions ROI + 14 autres
_ATLAS = Atlas(
    space="source",
    names=_NAMES,
    hemispheres=tuple(n.rsplit("-", 1)[1] for n in _NAMES),
    lobes=("frontal",) * len(_NAMES),
    coords_mm=np.zeros((len(_NAMES), 3)),
)


def _task_result(power: np.ndarray, labels: list[str], run: int = 3, condition: str = "mm_lr") -> RecordingResult:
    """power : (n_epochs, n_nodes) en log10, recopié sur les bandes alpha et bêta (bêta = 2 × alpha)."""
    n_epochs, n = power.shape
    return RecordingResult(
        dataset="test",
        subject=1,
        run=run,
        condition=condition,
        node_space="source",
        epoch_spec="task_lr",
        sfreq=160.0,
        n_times=560,
        node_names=_NAMES[:n],
        node_hemispheres=_ATLAS.hemispheres[:n],
        node_lobes=_ATLAS.lobes[:n],
        node_coords_mm=np.zeros((n, 3)),
        bands=("alpha", "beta"),
        metrics=("wpli",),
        conn_epochs=np.zeros((n_epochs, 1, 2, n, n), dtype=np.float32),
        power_epochs=np.stack([power, 2 * power], axis=1).astype(np.float32),
        epoch_onsets=np.arange(n_epochs, dtype=np.int64),
        epoch_labels=np.array(labels),
        n_epochs_total=n_epochs + 1,
    )


def test_erd_pools_task_labels_and_runs() -> None:
    a = _task_result(np.array([[1.0, 0.0], [0.0, 0.0], [3.0, 1.0]]), ["rest", "left", "right"])
    b = _task_result(np.array([[2.0, 0.0], [-3.0, 2.0]]), ["rest", "left"], run=7)
    # tâche : (0 + 3 - 3) / 3 = 0 et (0 + 1 + 2) / 3 = 1 ; rest : (1 + 2) / 2 = 1.5 et 0
    np.testing.assert_allclose(erd([a, b], "alpha"), [-1.5, 1.0])
    np.testing.assert_allclose(erd([a, b], "beta"), [-3.0, 2.0])


def test_erd_requires_both_conditions() -> None:
    with pytest.raises(ValueError):
        erd([_task_result(np.zeros((2, 2)), ["left", "right"])], "alpha")


def test_epoch_label_counts() -> None:
    r = _task_result(np.zeros((4, 2)), ["rest", "left", "rest", "right"])
    counts = epoch_label_counts({1: {"mm_lr": [r]}})
    assert counts[["rest", "left", "right", "n_epochs_total"]].iloc[0].tolist() == [2, 1, 1, 5]


def test_roi_top_check() -> None:
    values = np.arange(len(_NAMES), dtype=float)
    values[: len(MOTOR_ROI)] -= 100  # ROI = les 6 régions les plus négatives
    out = roi_top_check(values, _ATLAS, top=10)
    assert out["n_roi_in_top"] == 6
    assert out["roi_ranks"] == {r: k + 1 for k, r in enumerate(MOTOR_ROI)}
    assert out["passed"]
    out = roi_top_check(-values, _ATLAS, top=10)  # ROI en queue
    assert out["n_roi_in_top"] == 0
    assert out["hypergeom_p"] == pytest.approx(1.0)
    assert not out["passed"]


def test_roi_wilcoxon_tests() -> None:
    roi = -np.linspace(0.1, 1.0, 10)
    out = roi_erd_negative_test(roi)
    assert out["n_subjects_negative"] == 10
    assert out["wilcoxon_p_less"] == pytest.approx(1 / 2**10)  # loi exacte, n = 10
    assert out["passed"]
    c = mm_vs_mi_test(roi, roi / 2)  # MM deux fois plus négatif que MI chez tous les sujets
    assert c["n_subjects_mm_stronger"] == 10
    assert c["wilcoxon_p_less"] == pytest.approx(1 / 2**10)


def test_task_erd_report_structure() -> None:
    rng = np.random.default_rng(0)
    n = len(_NAMES)
    labels = ["rest", "left", "right"] * 4
    results: dict[int, dict[str, list[RecordingResult]]] = {}
    for s in range(1, 8):
        for cond, shift in (("mm_lr", -0.3), ("mi_lr", -0.1)):
            power = rng.normal(size=(12, n))
            power[[1, 2, 4, 5, 7, 8, 10, 11], : len(MOTOR_ROI)] += shift  # ERD sur la ROI
            results.setdefault(s, {})[cond] = [_task_result(power, labels, condition=cond)]
    erds = erd_by_subject(results)
    assert erds["mm_lr", "alpha"].shape == (7, n)
    assert roi_mean(erds["mm_lr", "alpha"], _ATLAS).shape == (7,)
    report = task_erd_report(erds, _ATLAS)
    assert set(report) == {
        "roi",
        "mm_lr_alpha",
        "mm_lr_beta",
        "mi_lr_alpha",
        "mi_lr_beta",
        "c_mm_vs_mi_alpha",
        "c_mm_vs_mi_beta",
    }


def test_cohort_erd_streams_from_cache(tmp_path: Path) -> None:
    paths = Paths(root=tmp_path)
    rng = np.random.default_rng(1)
    n = len(_NAMES)
    labels = ["rest", "left", "right"] * 3
    expected = {}
    for s in (1, 2, 3):
        for cond, run in (("mm_lr", 3), ("mi_lr", 4)):
            if s == 3 and cond == "mi_lr":
                continue  # sujet 3 sans MI : écarté
            r = replace(_task_result(rng.normal(size=(9, n)), labels, run=run, condition=cond), subject=s)
            save_result(r, result_path(paths.cache, "eegmmidb", "source", s, run, "task_lr"))
            expected[s, cond] = erd([r], "alpha")
    erds, atlas, kept = cohort_erd(paths, [1, 2, 3])
    assert kept == [1, 2]
    assert atlas.names == _NAMES
    np.testing.assert_allclose(
        erds["mi_lr", "alpha"], np.stack([expected[1, "mi_lr"], expected[2, "mi_lr"]]), rtol=1e-6
    )
