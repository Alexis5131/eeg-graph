from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from eeg_graph.config import Band
from eeg_graph.connectivity import (
    ComplexArray,
    FloatArray,
    aec_orth,
    band_analytic,
    ciplv,
    recording_connectivity,
    slice_epochs,
    wpli,
)

SFREQ = 160.0
ALPHA = Band("alpha", 8.0, 13.0)


def _narrowband(rng: np.random.Generator, n_samples: int) -> np.ndarray:
    return band_analytic(rng.standard_normal((1, n_samples)), SFREQ, Band("x", 6.0, 15.0)).real[0]


def _single_epoch(x: np.ndarray) -> ComplexArray:
    """(n_nodes, T) réel -> (1, n_nodes, T) analytique en alpha."""
    return band_analytic(x, SFREQ, ALPHA)[None]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def test_slice_epochs_shape_and_bounds() -> None:
    z = np.arange(2 * 100).reshape(2, 100)
    out = slice_epochs(z, np.array([0, 10, 50]), 20)
    assert out.shape == (3, 2, 20)
    np.testing.assert_array_equal(out[1, 0], np.arange(10, 30))
    with pytest.raises(ValueError):
        slice_epochs(z, np.array([90]), 20)


def test_wpli_lagged_coupling_is_one(rng: np.random.Generator) -> None:
    s = _narrowband(rng, 60 * int(SFREQ))
    # retard de 3 échantillons (18.75 ms) : déphasage de 54° à 88° dans 8-13 Hz, toujours du même signe
    x = np.vstack([s, np.roll(s, 3)]) + 0.05 * rng.standard_normal((2, s.size))
    w = wpli(_single_epoch(x))[0]
    assert w[0, 1] > 0.95
    assert w[0, 1] == pytest.approx(w[1, 0])
    assert np.all(np.diag(w) == 0)


def test_wpli_ignores_zero_lag_mixing(rng: np.random.Generator) -> None:
    """Un mélange instantané (conduction de volume, fuite de sources) ne doit pas gonfler le wPLI."""
    s = _narrowband(rng, 60 * int(SFREQ))
    x = np.vstack([s, s + 0.1 * _narrowband(rng, s.size)])
    assert wpli(_single_epoch(x))[0, 0, 1] < 0.15
    assert ciplv(_single_epoch(x))[0, 0, 1] < 0.15


def test_wpli_independent_noise_is_small(rng: np.random.Generator) -> None:
    x = rng.standard_normal((2, 60 * int(SFREQ)))
    assert wpli(_single_epoch(x))[0, 0, 1] < 0.15


def test_ciplv_lagged_coupling_is_high(rng: np.random.Generator) -> None:
    s = _narrowband(rng, 60 * int(SFREQ))
    x = np.vstack([s, np.roll(s, 3)]) + 0.05 * rng.standard_normal((2, s.size))
    c = ciplv(_single_epoch(x))[0]
    assert c[0, 1] > 0.9
    assert c.min() >= 0.0 and c.max() <= 1.0


def test_aec_orth_detects_shared_envelope(rng: np.random.Generator) -> None:
    n = 60 * int(SFREQ)
    t = np.arange(n) / SFREQ
    # modulation profonde : l'enveloppe propre (Rayleigh) de chaque porteuse dilue la corrélation
    envelope = np.exp(1.5 * np.sin(2 * np.pi * 0.2 * t))
    shared = np.vstack([envelope * _narrowband(rng, n), envelope * _narrowband(rng, n)])
    independent = np.vstack([_narrowband(rng, n), _narrowband(rng, n)])
    a_shared = aec_orth(_single_epoch(shared))[0]
    a_indep = aec_orth(_single_epoch(independent))[0]
    assert a_shared[0, 1] > 0.4
    assert a_indep[0, 1] < 0.2
    np.testing.assert_allclose(a_shared, a_shared.T)
    assert np.all(np.diag(a_shared) == 0)


def test_recording_connectivity_shapes(rng: np.random.Generator) -> None:
    n_nodes, n_times = 5, 4 * int(SFREQ)
    y = rng.standard_normal((n_nodes, 30 * int(SFREQ)))
    onsets = np.arange(0, 7) * n_times
    conn, power = recording_connectivity(y, SFREQ, onsets, n_times)
    assert conn.shape == (7, 3, 5, n_nodes, n_nodes)
    assert power.shape == (7, 5, n_nodes)
    assert conn.dtype == np.float32
    assert np.isfinite(conn).all()
    np.testing.assert_allclose(conn, np.swapaxes(conn, -1, -2), atol=1e-6)
    assert conn.min() >= 0.0 and conn.max() <= 1.0


@pytest.mark.parametrize("scale", [1e-11, 1e6])
def test_metrics_are_scale_invariant(rng: np.random.Generator, scale: float) -> None:
    """Régression : les séries de sources valent ~1e-11, un epsilon absolu annulait le wPLI."""
    s = _narrowband(rng, 20 * int(SFREQ))
    x = np.vstack([s, np.roll(s, 3), _narrowband(rng, s.size)]) + 0.05 * rng.standard_normal((3, s.size))
    z = _single_epoch(x)
    funcs: list[Callable[[ComplexArray], FloatArray]] = [wpli, ciplv, aec_orth]
    for func in funcs:
        np.testing.assert_allclose(func(z * scale), func(z), rtol=1e-6, atol=1e-9)
    assert wpli(z * scale)[0, 0, 1] > 0.9
