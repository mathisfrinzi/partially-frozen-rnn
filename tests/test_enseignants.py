"""Partie 9, étape 5 : les enseignants plantent bien leur structure, à la distance demandée de W₀."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")


def _m48():
    path = Path(__file__).resolve().parents[1] / "experiments" / "48_enseignants.py"
    spec = importlib.util.spec_from_file_location("exp48", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def W0():
    rng = np.random.default_rng(0)
    return rng.normal(size=(20, 20)) / np.sqrt(20)


@pytest.mark.parametrize("task", ["T1 tournée", "T2 cisaillée", "T3 accordée", "T4 bande", "T5 témoin"])
@pytest.mark.parametrize("amp", [0.1, 0.5])
def test_distance_is_the_requested_one(W0, task, amp):
    m = _m48()
    Wt = m.teacher(task, amp, W0, 0)
    assert np.linalg.norm(Wt - W0) / np.linalg.norm(W0) == pytest.approx(amp, rel=1e-4)


def test_rotation_keeps_the_spectrum(W0):
    m = _m48()
    Wt = m.teacher("T1 tournée", 0.5, W0, 0)
    assert np.allclose(np.sort(np.abs(np.linalg.eigvals(Wt))), np.sort(np.abs(np.linalg.eigvals(W0))), atol=1e-8)
    assert np.trace(Wt) == pytest.approx(np.trace(W0), abs=1e-10)


def test_tuned_keeps_the_singular_vectors(W0):
    m = _m48()
    Wt = m.teacher("T3 accordée", 0.5, W0, 0)
    U, s, Vt = np.linalg.svd(W0)
    D = U.T @ Wt @ Vt.T                                             # diagonale dans la base singulière de W₀
    assert np.allclose(D - np.diag(np.diag(D)), 0, atol=1e-10)


def test_shear_and_band_are_triangular_in_the_Q_basis(W0):
    m = _m48()
    Q, R = m.qr_pos(W0)
    S2 = Q.T @ m.teacher("T2 cisaillée", 0.5, W0, 0) - R
    assert np.allclose(np.tril(S2), 0, atol=1e-10) and np.abs(np.triu(S2, 11)).max() > 0
    S4 = Q.T @ m.teacher("T4 bande", 0.5, W0, 0) - R
    assert np.allclose(np.tril(S4), 0, atol=1e-10) and np.allclose(np.triu(S4, 11), 0, atol=1e-10)


def test_data_has_the_requested_noise_floor(W0):
    m = _m48()
    Win = np.random.default_rng(1).uniform(-0.5, 0.5, (20, 1))
    u, y, nt, Wt = m.make_data("T2 cisaillée", 0.1, Win, W0, 0)
    assert len(u) == len(y) == m.T_LEN and nt == int(0.75 * m.T_LEN)
    assert abs(y.mean()) < 0.05 and y.std() == pytest.approx(1.0, abs=0.01)
