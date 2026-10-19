"""Partie 9, étapes 2 et 3 : modèle à sous-espace appris, sondage en un point d'entraînement, choix de structure."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hfrnn import structure_signals as SS     # noqa: E402
from hfrnn import torch_models as TM          # noqa: E402

N = 12


@pytest.fixture(scope="module")
def res():
    return SS.reservoir(N, 1, 0)


def _load(name):
    path = Path(__file__).resolve().parents[1] / "experiments" / name
    spec = importlib.util.spec_from_file_location("exp_" + name[:2], path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_subspace_directions_and_positions(res):
    Win, W = res
    rng = np.random.default_rng(0)
    V = np.linalg.qr(rng.normal(size=(N * N, 5)))[0].T
    m = TM.build_model(Win, W, "subspace", 0, torch.float64, "cpu", V=V)
    assert np.allclose(m.weights()[1].detach().numpy(), W)                     # départ = réservoir
    with torch.no_grad():
        m.theta[2] = 0.3
    assert np.allclose((m.weights()[1].detach().numpy() - W).ravel(), 0.3 * V[2])
    idx = np.array([3, 17, 55])
    m2 = TM.build_model(Win, W, "subspace", 0, torch.float64, "cpu", idx=idx)
    with torch.no_grad():
        m2.theta[:] = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64)
    D = (m2.weights()[1].detach().numpy() - W).ravel()
    assert np.allclose(D[idx], [1.0, -2.0, 0.5]) and np.count_nonzero(D) == 3
    assert TM.n_trainable(m2) == 3 + Win.size + N                              # θ, W_in, biais
    with pytest.raises(ValueError):
        TM.build_model(Win, W, "subspace", 0, torch.float64, "cpu")


def test_subspace_gradient_is_projection_of_full_gradient(res):
    """∂L/∂θ = V·∂L/∂W : le sous-espace voit la projection du gradient complet."""
    Win, W = res
    rng = np.random.default_rng(1)
    u = rng.normal(size=300)
    y = np.roll(u, 2)
    V = np.linalg.qr(rng.normal(size=(N * N, 4)))[0].T
    w, c = SS.ridge_readout(u, y, 250, Win, W)
    gW = SS.window_grads(u, y, [10, 80], Win, W, w, c)[0].mean(0)
    m = TM.build_model(Win, W, "subspace", 0, torch.float64, "cpu", V=V)
    X = m(torch.as_tensor(np.stack([u[10:70], u[80:140]]), dtype=torch.float64).unsqueeze(-1))
    pred = X @ w + c
    Y = torch.as_tensor(np.stack([y[10:70], y[80:140]]), dtype=torch.float64)
    from hfrnn import core
    loss = ((pred[:, core.LOSS_WASH:] - Y[:, core.LOSS_WASH:]) ** 2).mean()
    loss.backward()
    assert np.allclose(m.theta.grad.numpy(), V @ gW, atol=1e-8)


def test_probe_works_at_a_trained_point(res):
    """Le sondage fonctionne en un point autre que le départ (W et biais modifiés), et ses signaux sont finis."""
    Win, W = res
    rng = np.random.default_rng(2)
    u = rng.normal(size=400)
    y = np.roll(u, 3)
    b = rng.normal(0, 0.1, N)
    W1 = W + 0.01 * rng.normal(size=W.shape)
    S = [("RNN", {"kind": "convex", "alpha": 1.0}), ("accordé", {"kind": "spectrum"}), ("tourné", {"kind": "rotation"})]
    out = SS.probe(u, y, 300, Win, W1, b, 0, S, B=24)
    assert set(out) == {"RNN", "accordé", "tourné"}
    for s in out.values():
        assert all(np.isfinite(v) for v in s.values())
    assert out["RNN"]["part captée"] == pytest.approx(1.0)
    assert 0 < out["accordé"]["part captée"] < 1


def test_gradient_window_bias_matters(res):
    """Le gradient est calculé au biais donné (il change quand le biais change)."""
    Win, W = res
    rng = np.random.default_rng(3)
    u = rng.normal(size=300)
    y = np.roll(u, 2)
    w, c = SS.ridge_readout(u, y, 250, Win, W)
    g0 = SS.window_grads(u, y, [10, 90], Win, W, w, c)[0]
    g1 = SS.window_grads(u, y, [10, 90], Win, W, w, c, b=np.full(N, 0.5))[0]
    assert not np.allclose(g0, g1)


def test_subspaces_have_the_requested_dimension_and_are_orthonormal():
    m46 = _load("46_sous_espace_appris.py")
    rng = np.random.default_rng(0)
    D = 64
    GA, GB = rng.normal(size=(40, D)), rng.normal(size=(12, D))
    sub = m46.subspaces(GA, GB, [5, 20], 0)
    assert len(sub) == 12
    for (meth, r), s in sub.items():
        if "V" in s:
            assert s["V"].shape == (r, D)
            assert np.allclose(s["V"] @ s["V"].T, np.eye(r), atol=1e-8)
        else:
            assert len(s["idx"]) == r == len(set(s["idx"].tolist()))


def test_gradient_positions_pick_the_largest_mean_gradients():
    m46 = _load("46_sous_espace_appris.py")
    rng = np.random.default_rng(1)
    GA = rng.normal(size=(200, 30)) * 0.1
    GA[:, [4, 9, 21]] += np.array([3.0, -2.0, 1.5])
    GB = rng.normal(size=(50, 30)) * 0.1
    GB[:, [4, 9, 21]] += np.array([3.0, -2.0, 1.5])
    sub = m46.subspaces(GA, GB, [3], 0)
    assert set(sub[("positions du gradient", 3)]["idx"].tolist()) == {4, 9, 21}
    # à accord : une direction dont le signe change entre entraînement et validation est écartée
    GB2 = GB.copy()
    GB2[:, 4] *= -1
    sub2 = m46.subspaces(GA, GB2, [3], 0)
    assert 4 not in sub2[("positions à accord", 3)]["idx"].tolist()


def test_dynamic_choice_and_interaction():
    m45 = _load("45_signaux_dynamiques.py")
    # deux tâches, trois graines, un seul t : le signal « s » prédit le gain exactement sur une tâche
    names = m45.NAMES
    cells = {}
    for task, best in (("A", 1), ("B", 3)):
        cells[task] = {0: {}}
        for seed in "012":
            cell = {}
            for i, s in enumerate(names):
                good = float(i == best)
                cell[s] = {"gain": -good, "signal sur bruit": good + 0.01 * i, "dimension": i + 1}
            cells[task][0][seed] = cell
    p = m45.choose(cells, ["A", "B"], 0, "signal sur bruit", names)
    assert p["A"]["choisie"] == names[1] and p["A"]["regret"] == 0
    assert p["B"]["choisie"] == names[3]
    p = m45.choose(cells, ["A", "B"], 0, "taille", names)            # la taille choisit toujours la plus grande
    assert p["A"]["choisie"] == names[-1] and p["A"]["regret"] > 0
    X = np.eye(5)[[0, 1, 2, 3, 4]] + 0.0       # cinq tâches, chacune a sa meilleure structure
    r, pv = m45.interaction(X, X.copy(), np.random.default_rng(0), n_perm=200)
    assert r == pytest.approx(1.0) and pv < 0.05


def test_thaw_keeps_frozen_coordinates_at_zero(res):
    """Dégel : une coordonnée ne bouge pas avant son pas de dégel, puis apprend."""
    Win, W = res
    rng = np.random.default_rng(5)
    u = rng.normal(size=400)
    y = np.roll(u, 2)
    V = np.linalg.qr(rng.normal(size=(N * N, 4)))[0].T
    m = TM.build_model(Win, W, "subspace", 0, torch.float32, "cpu", V=V, thaw=np.array([0, 0, 6, 6]))
    TM.train_regression(u, y, Win, W, 0, model=m, steps=5, n_train=300)
    th = m.theta.detach().numpy()
    assert np.all(th[2:] == 0) and np.any(th[:2] != 0)                 # 5 pas : le second bloc est encore gelé
    TM.train_regression(u, y, Win, W, 0, model=m, steps=5, n_train=300)
    assert np.any(m.theta.detach().numpy()[2:] != 0)                    # 10 pas au total : dégelé


def test_multiplicative_tangent():
    m47 = _load("47_protocole_lie.py")
    rng = np.random.default_rng(0)
    N_ = 6
    W0 = rng.normal(size=(N_, N_))
    G = rng.normal(size=(5, N_ * N_))
    X = m47.tangent_samples(G, W0, "M⁻¹G (ε = 0.01)")
    assert X.shape == G.shape
    sv = np.linalg.svd(W0, compute_uv=False)
    lam = (0.01 * sv[0]) ** 2
    want = np.linalg.solve(W0.T @ W0 + lam * np.eye(N_), W0.T @ G[0].reshape(N_, N_))
    assert np.allclose(X[0].reshape(N_, N_), want, atol=1e-8)
    assert np.allclose(m47.tangent_samples(G, W0, "G brut"), G)
    V = rng.normal(size=(3, N_ * N_))
    assert np.allclose(m47.to_weights(V, W0, "M⁻¹G (ε = 0.1)")[1].reshape(N_, N_), W0 @ V[1].reshape(N_, N_))
