"""Partie 9 : espaces tangents des structures, projections, et signaux."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hfrnn import structure_signals as SS     # noqa: E402

N = 12


@pytest.fixture(scope="module")
def res():
    return SS.reservoir(N, 1, 0)


def test_rotation_jacobian_matches_autograd(res):
    Win, W = res
    Ja = SS.tangent_jacobian(Win, W, 0, {"kind": "rotation"})
    from hfrnn import torch_models as TM
    m = TM.build_model(Win, W, "rotation", 0, torch.float64, "cpu")
    # direction S = E_01 : Ω = E_01 − E_10, dW = Ω W − W Ω
    eps = 1e-6
    with torch.no_grad():
        m.S[0, 1] += eps
    dW = (m.weights()[1].detach().numpy() - W) / eps
    assert np.allclose(dW.ravel(), Ja[:, 0], atol=1e-5)


def test_shear_basis_is_upper_triangle_in_Q(res):
    Win, W = res
    U = SS.orthonormal_basis(SS.tangent_jacobian(Win, W, 0, {"kind": "decomp", "form": "iwasawa_N"}))
    assert U.shape[1] == N * (N - 1) // 2
    Q, R = np.linalg.qr(W)
    G = np.random.default_rng(1).normal(size=(N, N))
    proj = (G.ravel() @ U) @ U.T
    want = (Q @ np.triu(Q.T @ G, 1)).ravel()                   # projection orthogonale sur Q·(strictement triangulaire)
    assert np.allclose(proj, want, atol=1e-8)


def test_accorde_coords_are_singular_diagonal(res):
    Win, W = res
    U = SS.orthonormal_basis(SS.tangent_jacobian(Win, W, 0, {"kind": "spectrum"}))
    Us, s, Vt = np.linalg.svd(W)
    G = np.random.default_rng(2).normal(size=(N, N))
    keep = s > 1e-10 * s.max()                                  # une direction de σ nul ne bouge pas avec g
    assert np.isclose(((G.ravel() @ U) ** 2).sum(), (np.diag(Us.T @ G @ Vt.T)[keep] ** 2).sum())


def test_full_and_empty_structures(res):
    Win, W = res
    assert SS.tangent_jacobian(Win, W, 0, {"kind": "convex", "alpha": 1.0}) == "full"
    assert SS.tangent_jacobian(Win, W, 0, {"kind": "spectrum", "learn_g": False}) is None
    gW, gI, gb = np.ones((3, N * N)), np.ones((3, N)), np.ones((3, N))
    assert SS.coords(gW, gI, gb, "full").shape == (3, N * N + 2 * N)
    assert SS.coords(gW, gI, gb, None).shape == (3, 2 * N)


def test_window_grads_match_finite_difference(res):
    Win, W = res
    u = np.sin(np.arange(300) * 0.3); y = np.roll(u, -2)
    w, c = SS.ridge_readout(u, y, 200, Win, W)
    gW, gI, gb = SS.window_grads(u, y, [10, 50], Win, W, w, c)
    eps = 1e-6
    W2 = W.copy(); W2[1, 2] += eps

    def L(Wm):
        import torch as T
        ut = T.as_tensor(u); U = ut[10:10 + 60].view(1, -1, 1)
        X = SS._run(U, T.as_tensor(Win), T.as_tensor(Wm), T.zeros(N, dtype=T.float64))[0]
        p = X @ w + c
        return float(((p[15:] - T.as_tensor(y[10:70])[15:]) ** 2).mean())
    assert np.isclose((L(W2) - L(W)) / eps, gW[0, 1 * N + 2], rtol=1e-3, atol=1e-8)


def test_signals_basic():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(64, 5)) + 3.0                          # une direction constante forte
    s = SS.signals(Z, Z + rng.normal(size=Z.shape) * 0.1, Z, lam=1.0, pcs=np.eye(5)[:2])
    assert 0.99 < s["part captée"] <= 1.0 + 1e-9 and s["signal sur bruit"] > 10 and s["accord entraînement-validation"] > 0.9
    assert 0 < s["pas de Newton"] and s["dimension"] == 5
