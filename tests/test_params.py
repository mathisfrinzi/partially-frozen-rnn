"""Tests des paramétrisations et de l'entraîneur."""
import numpy as np
import pytest

from hfrnn import core, params as P, tasks, runner


def _grad_check(param, seed=0, n=6):
    """Gradients des paramètres d'une paramétrisation contre des différences finies."""
    r = np.random.default_rng(seed)
    Win_r = r.normal(0, 0.5, (n, 1))
    W_r = r.normal(0, 0.3, (n, n)) * (r.random((n, n)) < 0.5)
    param.init(Win_r, W_r, np.random.default_rng(seed + 1))
    for p in param.params:                         # point générique (pas V = 0, etc.)
        p += r.normal(0, 0.05, p.shape)
    U, Y = r.normal(size=(3, 8)), r.normal(size=(3, 8))
    b, Wout, c = r.normal(0, 0.1, n), r.normal(0, 0.5, (n, 1)), np.zeros(1)

    def loss():
        Win_t, W_t = param.compose(Win_r, W_r)
        return core.bptt(U, Y, Win_t, W_t, b, Wout, c)[0]

    Win_t, W_t = param.compose(Win_r, W_r)
    _, dWin, dW, *_ = core.bptt(U, Y, Win_t, W_t, b, Wout, c)
    eps = 1e-6
    for p, g in zip(param.params, param.grads(dWin, dW, Win_r, W_r)):
        for _ in range(5):
            idx = tuple(r.integers(0, s) for s in p.shape)
            old = p[idx]
            p[idx] = old + eps; lp = loss()
            p[idx] = old - eps; lm = loss()
            p[idx] = old
            assert g[idx] == pytest.approx((lp - lm) / (2 * eps), rel=1e-4, abs=1e-9)


@pytest.mark.parametrize("make", [lambda: P.Convex(0.3), lambda: P.LearnedAlpha(0.3),
                                  lambda: P.LowRank(2), lambda: P.SparseMask(),
                                  lambda: P.SuperMask("sigmoid", win="free"),
                                  lambda: P.SuperMask("sigmoid", win="mask"),
                                  lambda: P.SuperMask("gain", win="mask"),
                                  lambda: P.Rotation("free"), lambda: P.Rotation("rotate"),
                                  lambda: P.SpectrumOnly(),
                                  lambda: P.Combo(spectrum=True, win="mask"),
                                  lambda: P.Combo(spectrum=True, mask=True),
                                  lambda: P.Combo(mask=True, slow=0.1),
                                  lambda: P.Combo(spectrum=True, mask=True, slow=0.1, win="mask")])
def test_parametrization_gradients(monkeypatch, make):
    monkeypatch.setattr(core, "LOSS_WASH", 2)
    _grad_check(make())


def test_starting_point_is_the_reservoir():
    """Au départ, toutes les paramétrisations donnent exactement le réservoir."""
    Win, W = core.make_reservoir(np.random.default_rng(0))
    for p in (P.Convex(0.1), P.LearnedAlpha(0.5), P.LowRank(4), P.SparseMask(), P.Rotation("rotate"),
              P.SpectrumOnly(), P.SuperMask("topk", keep=1.0, win="fixed")):
        p.init(Win, W, np.random.default_rng(1))
        Win_t, W_t = p.compose(Win, W)
        assert np.allclose(Win_t, Win) and np.allclose(W_t, W)


def test_sparse_update_stays_on_reservoir_graph():
    Win, W = core.make_reservoir(np.random.default_rng(0))
    p = P.SparseMask()
    p.init(Win, W, None)
    p.W_l += 1.0
    _, W_t = p.compose(Win, W)
    assert np.all((W_t != 0) == (W != 0))


def test_pure_reservoir_solves_fibonacci():
    u, y = tasks.load("Fibonacci mod 10")
    Win, W = core.make_reservoir(np.random.default_rng(0))
    assert P.pure_reservoir(u, y, Win, W) < 1e-6


def test_training_reduces_loss():
    u, y = tasks.load("NARMA10")
    Win, W = core.make_reservoir(np.random.default_rng(0))
    _, info = P.train(P.Convex(1.0), u, y, Win, W, seed=0, steps=60, log_every=1)
    loss = np.array(info["loss"])
    assert loss[-10:].mean() < loss[:10].mean()


def test_training_is_deterministic():
    u, y = tasks.load("Mackey-Glass (+15)")
    Win, W = core.make_reservoir(np.random.default_rng(3))
    a, _ = P.train(P.LowRank(2), u, y, Win, W, seed=3, steps=10)
    b, _ = P.train(P.LowRank(2), u, y, Win, W, seed=3, steps=10)
    assert a == b


def test_runner_spec_matches_direct_call():
    spec = {"kind": "convex", "alpha": 0.1, "task": "NARMA10", "seed": 0}
    u, y = tasks.load("NARMA10")
    Win, W = core.make_reservoir(np.random.default_rng(0))
    direct, _ = P.train(P.Convex(0.1), u, y, Win, W, seed=0, steps=5)
    old = core.STEPS
    try:
        core.STEPS = 5
        via_runner, _ = runner.run_one(spec)
    finally:
        core.STEPS = old
    assert via_runner == pytest.approx(direct, rel=1e-12)


def test_memory_task():
    u, y = tasks.task_delay(500, delay=15)
    assert np.allclose(y[15:], u[:-15])


def test_topk_supermask_keeps_requested_fraction():
    Win, W = core.make_reservoir(np.random.default_rng(0))
    p = P.SuperMask("topk", keep=0.3, win="fixed")
    p.init(Win, W, np.random.default_rng(1))
    m = p.mask()
    assert m[W == 0].sum() == 0                                   # jamais de connexion créée
    assert m[W != 0].mean() == pytest.approx(0.3, abs=0.01)
    _, W_t = p.compose(Win, W)
    assert set(np.unique(W_t[W != 0])) <= set(np.unique(W[W != 0])) | {0.0}   # valeurs inchangées


def test_rotation_preserves_spectrum():
    Win, W = core.make_reservoir(np.random.default_rng(0))
    p = P.Rotation("free")
    p.init(Win, W, None)
    p.S += np.random.default_rng(2).normal(0, 0.3, p.S.shape)
    _, W_t = p.compose(Win, W)
    ev = lambda M: np.sort_complex(np.round(np.linalg.eigvals(M), 8))    # noqa: E731
    assert np.allclose(ev(W_t), ev(W), atol=1e-6)
    assert np.allclose(np.linalg.svd(W_t, compute_uv=False), np.linalg.svd(W, compute_uv=False))
    assert not np.allclose(W_t, W)


def test_spectrum_only_starts_at_reservoir():
    Win, W = core.make_reservoir(np.random.default_rng(0))
    p = P.SpectrumOnly()
    p.init(Win, W, None)
    assert np.allclose(p.compose(Win, W)[1], W)


def test_dense_reservoir_has_requested_density():
    _, W = core.make_reservoir(np.random.default_rng(0), sparsity=1.0)
    assert np.count_nonzero(W) == W.size
    assert core.spectral_radius(W) == pytest.approx(core.RHO)


def test_combo_reduces_to_parts():
    """Une combinaison avec une seule composante redonne le modèle de la partie correspondante."""
    Win, W = core.make_reservoir(np.random.default_rng(0))
    r = np.random.default_rng(1)
    a, b = P.Combo(spectrum=True), P.SpectrumOnly()
    a.init(Win, W, r); b.init(Win, W, r)
    a.g += 0.1 * r.normal(size=a.g.shape); b.g[:] = a.g
    assert np.allclose(a.compose(Win, W)[1], b.compose(Win, W)[1])
    a, b = P.Combo(mask=True, win="mask"), P.SuperMask("sigmoid", win="mask")
    a.init(Win, W, r); b.init(Win, W, r)
    a.s += r.normal(size=a.s.shape); b.s[:] = a.s
    assert np.allclose(a.compose(Win, W)[1], b.compose(Win, W)[1])
    assert np.allclose(a.compose(Win, W)[0], b.compose(Win, W)[0])
