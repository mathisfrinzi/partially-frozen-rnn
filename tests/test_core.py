"""Tests des briques de base : BPTT, ridge, Adam, réservoir, tâches."""
import numpy as np
import pytest

from hfrnn import core, tasks


# ----------------------------------------------------------------------
# BPTT : gradient analytique vs différences finies
# ----------------------------------------------------------------------
def _small_problem(monkeypatch, n=6, batch=3, length=8, seed=0):
    monkeypatch.setattr(core, "N", n)
    monkeypatch.setattr(core, "LOSS_WASH", 2)
    r = np.random.default_rng(seed)
    params = dict(
        Win=r.normal(0, 0.5, (n, 1)),
        W=r.normal(0, 0.4, (n, n)),
        b=r.normal(0, 0.1, n),
        Wout=r.normal(0, 0.5, (n, 1)),
        c=r.normal(0, 0.1, 1),
    )
    U = r.normal(size=(batch, length))
    Y = r.normal(size=(batch, length))
    return params, U, Y


def test_bptt_gradients_match_finite_differences(monkeypatch):
    params, U, Y = _small_problem(monkeypatch)
    names = ["Win", "W", "b", "Wout", "c"]

    def loss_of(p):
        return core.bptt(U, Y, p["Win"], p["W"], p["b"], p["Wout"], p["c"])[0]

    out = core.bptt(U, Y, params["Win"], params["W"], params["b"], params["Wout"], params["c"])
    grads = dict(zip(names, out[1:]))
    rng = np.random.default_rng(1)
    eps = 1e-6
    for name in names:
        P = params[name]
        for _ in range(6):
            idx = tuple(rng.integers(0, s) for s in P.shape)
            p_plus = {k: v.copy() for k, v in params.items()}
            p_minus = {k: v.copy() for k, v in params.items()}
            p_plus[name][idx] += eps
            p_minus[name][idx] -= eps
            numeric = (loss_of(p_plus) - loss_of(p_minus)) / (2 * eps)
            assert grads[name][idx] == pytest.approx(numeric, rel=1e-4, abs=1e-8), (name, idx)


def test_alpha_chain_rule_matches_finite_differences(monkeypatch):
    """dLoss/dalpha = somme(G * (ΔW - W₀)) pour W = (1-alpha) W₀ + alpha ΔW."""
    params, U, Y = _small_problem(monkeypatch, seed=3)
    r = np.random.default_rng(4)
    Win_r, W_r = params["Win"], params["W"]
    Win_l = Win_r + r.normal(0, 0.2, Win_r.shape)
    W_l = W_r + r.normal(0, 0.2, W_r.shape)

    def total(alpha):
        return (1 - alpha) * Win_r + alpha * Win_l, (1 - alpha) * W_r + alpha * W_l

    def loss_at(alpha):
        Win_t, W_t = total(alpha)
        return core.bptt(U, Y, Win_t, W_t, params["b"], params["Wout"], params["c"])[0]

    alpha, eps = 0.4, 1e-6
    Win_t, W_t = total(alpha)
    _, dWin, dW, *_ = core.bptt(U, Y, Win_t, W_t, params["b"], params["Wout"], params["c"])
    analytic = (dWin * (Win_l - Win_r)).sum() + (dW * (W_l - W_r)).sum()
    numeric = (loss_at(alpha + eps) - loss_at(alpha - eps)) / (2 * eps)
    assert analytic == pytest.approx(numeric, rel=1e-4)


# ----------------------------------------------------------------------
# Réservoir, ridge, Adam
# ----------------------------------------------------------------------
def test_reservoir_has_requested_spectral_radius_and_sparsity():
    Win, W = core.make_reservoir(np.random.default_rng(0))
    assert Win.shape == (core.N, 1) and W.shape == (core.N, core.N)
    assert core.spectral_radius(W) == pytest.approx(core.RHO, abs=1e-9)
    density = np.count_nonzero(W) / W.size
    assert density == pytest.approx(core.SPARSITY, abs=0.03)


def test_run_states_starts_from_zero_state():
    Win, W = core.make_reservoir(np.random.default_rng(0))
    u = np.array([0.5, -0.3, 0.8])
    X = core.run_states(u, Win, W, np.zeros(core.N))
    assert X.shape == (3, core.N)
    assert np.allclose(X[0], np.tanh(Win[:, 0] * u[0]))
    assert np.allclose(X[1], np.tanh(Win[:, 0] * u[1] + W @ X[0]))


def test_ridge_recovers_linear_map():
    r = np.random.default_rng(0)
    X = r.normal(size=(500, core.N))
    true_w = r.normal(size=core.N + 1)
    y = np.hstack([X, np.ones((500, 1))]) @ true_w
    w = core.ridge_fit(X, y, lam=1e-9)
    assert np.allclose(w, true_w, atol=1e-5)
    assert core.nmse(core.ridge_predict(X, w), y) < 1e-10


def test_adam_minimises_a_quadratic():
    x = np.array([5.0, -3.0])
    opt = core.Adam([x], lr=0.1)
    for _ in range(500):
        opt.step([2 * x])          # gradient de ||x||^2
    assert np.linalg.norm(x) < 1e-2


# ----------------------------------------------------------------------
# Tâches
# ----------------------------------------------------------------------
def test_fibonacci_task_is_fibonacci_mod_10():
    u, y = tasks.task_fib(200)
    assert set(np.unique(u)) <= set(range(10))
    assert list(u[:10]) == [0, 1, 1, 2, 3, 5, 8, 3, 1, 4]
    assert np.all(y[1:] == (u[1:] + u[:-1]) % 10)       # u(t+1) = u(t) + u(t-1) mod 10
    assert np.all(u[1:] == y[:-1])                       # la cible est l'entrée décalée


def test_narma10_follows_its_definition():
    u, y = tasks.task_narma10(300)
    for t in (20, 100, 250):
        expected = (0.3 * y[t - 1] + 0.05 * y[t - 1] * y[t - 10:t].sum()
                    + 1.5 * u[t - 9] * u[t] + 0.1)
        assert y[t] == pytest.approx(expected)
    assert np.all(np.isfinite(y))


def test_mackey_glass_target_is_shifted_input():
    h = 15
    u, y = tasks.task_mackey_glass(300, horizon=h)
    assert np.allclose(y[:-h], u[h:])
    assert 0 < u.min() and u.max() < 2.0                 # reste borné, positif
