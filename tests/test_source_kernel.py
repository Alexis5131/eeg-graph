"""V5 : W @ X doit reproduire apply_inverse_epochs + extract_label_time_course (test d'intégration).

Nécessite les données S001 R01 et fsaverage (téléchargées en exécutant le notebook).
"""

from __future__ import annotations

import pytest

from eeg_graph.config import InverseConfig, Paths, PipelineConfig
from eeg_graph.loading import load_eegmmidb_run
from eeg_graph.preprocessing import filter_raw, set_average_reference
from eeg_graph.validation import kernel_equivalence

_PATHS = Paths()
pytestmark = pytest.mark.skipif(
    not (_PATHS.subjects_dir / "fsaverage" / "bem").exists() or not any(_PATHS.eegmmidb.rglob("S001R01.edf")),
    reason="données S001 / fsaverage absentes",
)


@pytest.mark.parametrize("method", ["eLORETA", "dSPM"])
def test_kernel_matches_mne_chain(method: str) -> None:
    raw = set_average_reference(filter_raw(load_eegmmidb_run(1, 1, _PATHS), PipelineConfig().preproc))
    cfg = InverseConfig(method=method)  # type: ignore[arg-type]  # littéral validé par le paramétrage
    assert kernel_equivalence(raw, cfg, _PATHS) < 1e-6
