from __future__ import annotations

import mne
import numpy as np
import pytest

from eeg_graph.config import REST_4S, TASK_LR, EEGMMIDBSpec
from eeg_graph.preprocessing import EpochingError, epoch_onsets

SFREQ = 160.0


def _raw(duration: float, annotations: list[tuple[float, float, str]], first_samp: int = 0) -> mne.io.RawArray:
    """Raw synthétique 2 canaux avec annotations (onset, durée, description), onsets relatifs au début."""
    info = mne.create_info(["C3", "C4"], SFREQ, "eeg")
    raw = mne.io.RawArray(np.zeros((2, round(duration * SFREQ))), info, first_samp=first_samp, verbose="error")
    onsets, durations, descriptions = zip(*annotations, strict=True)
    raw.set_annotations(mne.Annotations(list(onsets), list(durations), list(descriptions)), verbose="error")
    return raw


# Séquence type d'un run de tâche EEGMMIDB : T0 (4.2 s) et T1/T2 (4.1 s) alternés.
_SEQUENCE = [
    (0.0, 4.2, "T0"),
    (4.2, 4.1, "T2"),
    (8.3, 4.2, "T0"),
    (12.5, 4.1, "T1"),
    (16.6, 4.2, "T0"),
    (20.8, 4.1, "T1"),  # fenêtre 21.3-24.8 s : déborde du signal de 23 s, donc écartée
]


@pytest.mark.parametrize("first_samp", [0, 100])
def test_epoch_onsets_task_lr(first_samp: int) -> None:
    onsets, labels, n_times = epoch_onsets(_raw(23.0, _SEQUENCE, first_samp), TASK_LR)
    assert n_times == round(3.5 * SFREQ)
    expected = [round((t + 0.5) * SFREQ) for t, _, _ in _SEQUENCE[:-1]]
    np.testing.assert_array_equal(onsets, expected)
    assert labels.tolist() == ["rest", "right", "rest", "left", "rest"]
    # aucune fenêtre ne dépasse l'annotation suivante
    next_onsets = [round(t * SFREQ) for t, _, _ in _SEQUENCE[1:-1]]
    assert all(o + n_times <= nxt for o, nxt in zip(onsets[:-1], next_onsets, strict=True))


def test_epoch_onsets_ignores_unknown_annotations() -> None:
    raw = _raw(10.0, [(0.0, 4.2, "BAD_x"), (1.0, 4.1, "T1")])
    onsets, labels, _ = epoch_onsets(raw, TASK_LR)
    assert labels.tolist() == ["left"]
    assert onsets.tolist() == [round(1.5 * SFREQ)]


def test_epoch_onsets_without_task_annotations() -> None:
    with pytest.raises(EpochingError):
        epoch_onsets(_raw(10.0, [(0.0, 1.0, "BAD_x")]), TASK_LR)


def test_epoch_spec_routing() -> None:
    spec = EEGMMIDBSpec()
    assert all(spec.epoch_spec(run) is TASK_LR for run in (3, 4, 7, 8, 11, 12))
    assert spec.epoch_spec(1) is REST_4S and spec.epoch_spec(2) is REST_4S
    with pytest.raises(ValueError):
        spec.epoch_spec(5)
