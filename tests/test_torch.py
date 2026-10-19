"""La version PyTorch doit reproduire exactement la version numpy (float64, CPU).

Ignorés automatiquement si PyTorch n'est pas installé (ex. : CI légère).
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hfrnn import core, params as P, tasks  # noqa: E402
from hfrnn import torch_models as TM  # noqa: E402

STEPS = 15


@pytest.fixture
def data():
    u, y = tasks.load("NARMA10")
    Win, W = core.make_reservoir(np.random.default_rng(0))
    return u, y, Win, W


def test_reservoir_generator_matches_numpy():
    a = core.make_reservoir(np.random.default_rng(3))
    b = TM.make_reservoir(core.N, seed=3)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])


def test_forward_matches_run_states(data):
    u, y, Win, W = data
    m = TM.MixRNN(Win, W, kind="reservoir", dtype=torch.float64)
    X = m(torch.as_tensor(u[:200]).view(1, -1, 1))[0].numpy()
    assert np.allclose(X, core.run_states(u[:200], Win, W, np.zeros(core.N)), atol=1e-12)


def test_gradients_match_numpy_bptt():
    r = np.random.default_rng(0)
    n = 8
    Win, W = r.normal(0, .5, (n, 1)), r.normal(0, .3, (n, n))
    U, Y = r.normal(size=(3, 20)), r.normal(size=(3, 20))
    m = TM.MixRNN(Win, W, kind="convex", alpha=1.0, dtype=torch.float64)
    Wout = torch.tensor(r.normal(0, .5, (n, 1)), requires_grad=True)
    c = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    X = m(torch.as_tensor(U).unsqueeze(-1))
    pred = (X @ Wout)[..., 0] + c
    loss = ((pred[:, core.LOSS_WASH:] - torch.as_tensor(Y)[:, core.LOSS_WASH:]) ** 2).mean()
    loss.backward()
    ref = core.bptt(U, Y, Win, W, np.zeros(n), Wout.detach().numpy(), np.zeros(1))
    assert loss.item() == pytest.approx(ref[0])
    assert np.allclose(m.Win_l.grad.numpy(), ref[1]) and np.allclose(m.W_l.grad.numpy(), ref[2])
    assert np.allclose(m.bias.grad.numpy(), ref[3]) and np.allclose(Wout.grad.numpy(), ref[4])


@pytest.mark.parametrize("kind,kw,numpy_param", [
    ("convex", {"alpha": 1.0}, lambda: P.Convex(1.0)),
    ("convex", {"alpha": 0.2}, lambda: P.Convex(0.2)),
    ("convex", {"alpha": 1.0, "lr_mult": 0.1}, lambda: P.Convex(1.0, lr_mult=0.1)),
    ("convex", {"alpha": 1.0, "lr_mult": 0.1, "lr_mult_in": 1.0}, lambda: P.Convex(1.0, lr_mult=0.1, lr_mult_in=1.0)),
    ("lowrank", {"rank": 3}, lambda: P.LowRank(3)),
    ("sparse", {}, lambda: P.SparseMask()),
    ("combo", {"spectrum": True, "win": "mask"}, lambda: P.Combo(spectrum=True, win="mask")),
    ("combo", {"spectrum": True, "mask": True, "slow": 0.1}, lambda: P.Combo(spectrum=True, mask=True, slow=0.1)),
    ("combo", {"mask": True, "slow": 0.1, "win": "mask"}, lambda: P.Combo(mask=True, slow=0.1, win="mask")),
    ("supermask", {"mode": "sigmoid", "win": "free", "mult": 10.0}, lambda: P.SuperMask("sigmoid", win="free")),
    ("supermask", {"mode": "sigmoid", "win": "mask", "mult": 10.0}, lambda: P.SuperMask("sigmoid", win="mask")),
    ("supermask", {"mode": "topk", "keep": 0.5, "win": "free", "mult": 10.0, "rescale": True},
     lambda: P.SuperMask("topk", keep=0.5, win="free", rescale=True)),
    ("rotation", {"win": "free"}, lambda: P.Rotation("free")),
    ("rotation", {"win": "rotate"}, lambda: P.Rotation("rotate")),
    ("spectrum", {}, lambda: P.SpectrumOnly()),
])
def test_training_matches_numpy(data, kind, kw, numpy_param):
    u, y, Win, W = data
    ref, _ = P.train(numpy_param(), u, y, Win, W, seed=0, steps=STEPS)
    got, _ = TM.train_regression(u, y, Win, W, seed=0, kind=kind, steps=STEPS, dtype=torch.float64, **kw)
    assert got == pytest.approx(ref, rel=1e-5)


@pytest.mark.parametrize("init", ["ridge", 1.0])
def test_readout_init_matches_numpy(data, init):
    u, y, Win, W = data
    ref, _ = P.train(P.Convex(0.1), u, y, Win, W, seed=0, steps=STEPS, init=init)
    got, _ = TM.train_regression(u, y, Win, W, seed=0, kind="convex", alpha=0.1, steps=STEPS,
                                 dtype=torch.float64, init=init)
    assert got == pytest.approx(ref, rel=1e-5)


def test_reservoir_matches_numpy(data):
    u, y, Win, W = data
    got, _ = TM.train_regression(u, y, Win, W, seed=0, kind="reservoir", dtype=torch.float64)
    assert got == pytest.approx(P.pure_reservoir(u, y, Win, W), rel=1e-8)


def test_learned_alpha_matches_numpy(data):
    u, y, Win, W = data
    ref, ri = P.train(P.LearnedAlpha(0.3, lr_mult_theta=0.05 / core.LR), u, y, Win, W, seed=0, steps=STEPS)
    got, info = TM.train_regression(u, y, Win, W, seed=0, kind="lalpha", steps=STEPS, dtype=torch.float64,
                                    alpha0=0.3, lr_theta=0.05)
    assert got == pytest.approx(ref, rel=1e-5)
    assert info["alpha_rec"] == pytest.approx(ri["alpha_rec"], rel=1e-6)


# ----------------------------------------------------------------------
# Partie 5 : modèles existants
# ----------------------------------------------------------------------
from hfrnn.torch_baselines import Baseline  # noqa: E402


def test_lru_matches_complex_recurrence():
    """Le LRU écrit en réel doit reproduire la récurrence complexe h = λ h + γ B u."""
    full = Baseline("lru", n=6, n_in=1, seed=1, dtype=torch.float64)
    m = full.layers[0]
    U = torch.as_tensor(np.random.default_rng(0).normal(size=(2, 15, 1)))
    out = full(U).detach().numpy()
    lam = np.exp(-np.exp(m.nu_log.detach().numpy()) + 1j * np.exp(m.theta_log.detach().numpy()))
    g = np.exp(m.gamma_log.detach().numpy())
    Bc = (m.B_re.detach().numpy() + 1j * m.B_im.detach().numpy()) * g[:, None]
    Cc = m.C_re.detach().numpy() + 1j * m.C_im.detach().numpy()
    D = m.D.detach().numpy()
    for b in range(2):
        h = np.zeros(3, complex)
        for t in range(15):
            u = U[b, t].numpy()
            h = lam * h + Bc @ u
            y = (Cc @ h).real + D @ u
            ref = torch.nn.functional.gelu(torch.as_tensor(y)).numpy()
            assert np.allclose(out[b, t], ref)
    assert np.all(np.abs(lam) < 1)


@pytest.mark.parametrize("kind,kw", [("gru", {}), ("lstm", {}), ("lru", {}), ("gru", {"rec_init": "ortho"}),
                                     ("lstm", {"rec_init": "ortho"}), ("lru", {"layers": 2})])
def test_baselines_train_and_report_checkpoints(data, kind, kw):
    u, y, Win, W = data
    s0, info0 = TM.train_regression(u, y, Win, W, 0, kind=kind, steps=0, dtype=torch.float64, **kw)
    s, info = TM.train_regression(u, y, Win, W, 0, kind=kind, steps=40, dtype=torch.float64, eval_at=(20,), **kw)
    assert np.isfinite(s) and s != s0                      # l'entraînement change bien le modèle
    if not kw:
        assert s < s0                                      # et l'améliore (initialisation par défaut)
    assert set(info["checkpoints"]) == {20, 40} and info["checkpoints"][40] == s
    assert info["n_params"] > 0


def test_checkpoints_do_not_change_training(data):
    """Mesurer la NMSE en cours de route ne doit pas modifier l'entraînement."""
    u, y, Win, W = data
    a, _ = TM.train_regression(u, y, Win, W, 0, kind="spectrum", steps=STEPS, dtype=torch.float64)
    b, info = TM.train_regression(u, y, Win, W, 0, kind="spectrum", steps=STEPS, dtype=torch.float64, eval_at=(5,))
    assert a == b and 5 in info["checkpoints"]


def test_parameter_counts():
    n = 10
    Win, W = TM.make_reservoir(n, sparsity=0.5, seed=0)
    count = lambda k, **kw: TM.n_trainable(TM.build_model(Win, W, k, 0, torch.float64, "cpu", **kw))  # noqa: E731
    assert count("gru") == 3 * (n * 1 + n * n + 2 * n)
    assert count("lstm") == 4 * (n * 1 + n * n + 2 * n)
    assert count("spectrum") == n + n + n                  # g, W_in, biais
    assert count("reservoir") == 0


def test_careful_init():
    m = Baseline("lstm", n=8, rec_init="ortho", dtype=torch.float64)
    W = m.rnn.weight_hh_l0.detach().numpy()
    for k in range(4):
        Q = W[k * 8:(k + 1) * 8]
        assert np.allclose(Q @ Q.T, np.eye(8), atol=1e-10)
    assert np.allclose(m.rnn.bias_ih_l0.detach().numpy()[8:16], 1.0)


# ----------------------------------------------------------------------
# Partie 4.2 : améliorations du réservoir accordé
# ----------------------------------------------------------------------
from hfrnn.torch_tuned import Tuned  # noqa: E402


def test_tuned_default_is_the_tuned_reservoir(data):
    """Sans option, « tuned » est exactement le réservoir accordé de la partie 3."""
    u, y, Win, W = data
    a, _ = TM.train_regression(u, y, Win, W, 0, kind="spectrum", steps=STEPS, dtype=torch.float64)
    b, _ = TM.train_regression(u, y, Win, W, 0, kind="tuned", steps=STEPS, dtype=torch.float64)
    assert b == pytest.approx(a, rel=1e-8)


@pytest.mark.parametrize("kw", [{}, {"basis": "eig"}, {"gmax": 2.0}, {"gmax": 3.0}, {"poly": 4}, {"rank": 3},
                                {"basis": "eig", "poly": 3}])
def test_tuned_starts_at_the_reservoir(kw):
    Win, W = core.make_reservoir(np.random.default_rng(0))
    m = Tuned(Win, W, dtype=torch.float64, **kw)
    Win_t, W_t = m.weights()
    assert np.allclose(W_t.detach().numpy(), W, atol=1e-9) and np.allclose(Win_t.detach().numpy(), Win)


def test_tuned_eig_stays_real_and_keeps_eigenvectors():
    Win, W = core.make_reservoir(np.random.default_rng(1))
    m = Tuned(Win, W, basis="eig", dtype=torch.float64)
    with torch.no_grad():
        m.g.copy_(torch.rand(len(m.g), dtype=torch.float64) + 0.5)
        m.phi.copy_(torch.randn(len(m.phi), dtype=torch.float64) * 0.3)
    _, Wt = m.weights()
    Wt = Wt.detach().numpy()
    lam_new = np.linalg.eigvals(Wt)
    # les modules des valeurs propres ont bien changé, et W est réelle
    assert np.isrealobj(Wt) and not np.allclose(np.sort(np.abs(lam_new)), np.sort(np.abs(np.linalg.eigvals(W))))


@pytest.mark.parametrize("kind,kw", [("tuned", {"leak": True}), ("tuned", {"basis": "eig"}), ("tuned", {"gmax": 2.0}),
                                     ("tuned", {"poly": 4}), ("tuned", {"rank": 4}), ("deep_tuned", {})])
def test_tuned_variants_train(data, kind, kw):
    u, y, Win, W = data
    s0, _ = TM.train_regression(u, y, Win, W, 0, kind=kind, steps=0, dtype=torch.float64, **kw)
    s, info = TM.train_regression(u, y, Win, W, 0, kind=kind, steps=30, dtype=torch.float64, **kw)
    assert np.isfinite(s) and s < s0
    if kw.get("poly"):
        assert info["n_params"] == 4 + core.N + core.N        # coefficients, W_in, biais


# ----------------------------------------------------------------------
# Nouvelles variantes (parties 3, 4.2 et 6)
# ----------------------------------------------------------------------
from hfrnn.torch_tuned import FastTuned  # noqa: E402


@pytest.mark.parametrize("kw", [{"selective": True}, {"neuron_gains": True}, {"rotations": 10}, {"topk": 20},
                                {"selective": True, "neuron_gains": True, "rotations": 5, "rank": 2}])
def test_new_options_start_at_the_reservoir(kw):
    Win, W = core.make_reservoir(np.random.default_rng(0))
    m = Tuned(Win, W, dtype=torch.float64, **kw)
    assert np.allclose(m.weights()[1].detach().numpy(), W, atol=1e-9)
    ref = Tuned(Win, W, dtype=torch.float64)
    U = torch.as_tensor(np.random.default_rng(1).normal(size=(3, 12, 1)))
    assert torch.allclose(m(U), ref(U), atol=1e-10)


def test_factored_step_matches_matrix():
    """Le calcul direction par direction (rotations, gains par neurone, rang) donne bien W·x."""
    Win, W = core.make_reservoir(np.random.default_rng(2))
    m = Tuned(Win, W, dtype=torch.float64, neuron_gains=True, rotations=8, rank=3)
    r = np.random.default_rng(3)
    with torch.no_grad():
        for p in (m.g, m.dL, m.dR, m.theta, m.B):
            p.add_(torch.as_tensor(r.normal(0, 0.2, p.shape)))
    x = torch.as_tensor(r.normal(size=(4, core.N)))
    u = torch.zeros(4, 1, dtype=torch.float64)
    Wt = m.weights()[1]
    assert torch.allclose(m.recurrent_step(x, u), x @ Wt.T, atol=1e-10)


def test_selective_gains_depend_on_input():
    Win, W = core.make_reservoir(np.random.default_rng(2))
    m = Tuned(Win, W, dtype=torch.float64, selective=True)
    with torch.no_grad():
        m.G.add_(0.5)
    x = torch.ones(2, core.N, dtype=torch.float64)
    u = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    r = m.recurrent_step(x, u)
    assert torch.allclose(r[0], x[0] @ m.weights()[1].T) and not torch.allclose(r[0], r[1])


@pytest.mark.parametrize("spectrum", ["flat", "spread"])
def test_spectrum_controls_keep_directions_and_radius(spectrum):
    Win, W = core.make_reservoir(np.random.default_rng(0))
    m = Tuned(Win, W, dtype=torch.float64, spectrum=spectrum)
    Wt = m.weights()[1].detach().numpy()
    assert np.max(np.abs(np.linalg.eigvals(Wt))) == pytest.approx(core.RHO, rel=1e-6)
    U_, _, Vt = np.linalg.svd(W)
    assert np.allclose(np.abs(np.diag(U_.T @ Wt @ Vt.T)), m.sig.numpy(), atol=1e-8)


@pytest.mark.parametrize("n", [64, 101])
def test_fast_tuned_matches_dense_matrix(n):
    Win = np.random.default_rng(0).uniform(-0.5, 0.5, (n, 1))
    m = FastTuned(Win, seed=1, dtype=torch.float64)
    with torch.no_grad():
        m.g.mul_(torch.as_tensor(np.random.default_rng(2).uniform(0.5, 1.5, m.g.shape)))
    lam = (m.lam * m.g).detach().numpy()
    Wd = FastTuned.dense(lam, m.p1.numpy(), m.p2.numpy(), n)
    x = torch.as_tensor(np.random.default_rng(3).normal(size=(3, n)))
    assert np.allclose(m.matvec(x).detach().numpy(), x.numpy() @ Wd.T, atol=1e-10)
    sv = np.sort(np.linalg.svd(Wd, compute_uv=False))
    ref = np.sort(np.concatenate([np.abs(lam), np.abs(lam[1:(n + 1) // 2])]))     # paires conjuguées
    assert np.allclose(sv, ref, atol=1e-8)


def test_fast_tuned_radius():
    m = FastTuned(np.zeros((200, 1)), seed=0, dtype=torch.float64)
    Wd = FastTuned.dense(m.lam.numpy(), m.p1.numpy(), m.p2.numpy(), 200)
    assert np.max(np.abs(np.linalg.eigvals(Wd))) == pytest.approx(core.RHO, rel=1e-6)


def test_dual_ridge_equals_primal():
    r = np.random.default_rng(0)
    X, y = torch.as_tensor(r.normal(size=(30, 50))), torch.as_tensor(r.normal(size=30))
    w = TM.ridge(X, y, lam=1e-2)                                        # forme duale (51 > 30)
    A = torch.cat([X, torch.ones(30, 1, dtype=X.dtype)], 1)
    w2 = torch.linalg.solve(A.T @ A + 1e-2 * torch.eye(51, dtype=X.dtype), A.T @ y)
    assert torch.allclose(w, w2, atol=1e-8)


@pytest.mark.parametrize("kind,kw", [("tuned", {"selective": True}), ("tuned", {"neuron_gains": True}),
                                     ("tuned", {"penalty": 1e-3}), ("tuned", {"rotations": 16}),
                                     ("tuned", {"spectrum": "flat"}), ("tuned", {"topk": 10}),
                                     ("tuned", {"topk": 10, "select": "random"}),
                                     ("fast_tuned", {})])
def test_new_variants_train(data, kind, kw):
    u, y, Win, W = data
    s0, _ = TM.train_regression(u, y, Win, W, 0, kind=kind, steps=0, dtype=torch.float64, **kw)
    s, _ = TM.train_regression(u, y, Win, W, 0, kind=kind, steps=30, dtype=torch.float64, **kw)
    assert np.isfinite(s) and s < s0


def test_fast_reservoir_is_untrained(data):
    u, y, Win, W = data
    a, info = TM.train_regression(u, y, Win, W, 0, kind="fast_reservoir", dtype=torch.float64)
    assert np.isfinite(a) and info["n_params"] == 0


# ----------------------------------------------------------------------
# Partie 5 : réservoir accordé à portes
# ----------------------------------------------------------------------
from hfrnn.torch_tuned import GatedTuned  # noqa: E402


@pytest.mark.parametrize("cell", ["gru", "gru_light", "mingru"])
def test_gated_starts_near_the_tuned_reservoir(cell):
    """Au départ, une cellule à portes (portes presque ouvertes) se comporte presque comme le réservoir accordé."""
    Win, W = core.make_reservoir(np.random.default_rng(0))
    U = torch.as_tensor(np.random.default_rng(1).normal(size=(2, 40, 1)))
    a = GatedTuned(Win, W, dtype=torch.float64, cell=cell)(U)
    b = Tuned(Win, W, dtype=torch.float64)(U)
    assert (a - b).abs().max() < 0.15 and (a - b).abs().mean() < 0.03


def test_gated_gru_without_gates_is_the_tuned_reservoir():
    """Portes forcées complètement ouvertes : exactement le réservoir accordé."""
    Win, W = core.make_reservoir(np.random.default_rng(0))
    m = GatedTuned(Win, W, dtype=torch.float64, cell="gru")
    with torch.no_grad():
        m.c_z.fill_(50.0); m.c_r.fill_(50.0)
    U = torch.as_tensor(np.random.default_rng(1).normal(size=(2, 30, 1)))
    assert torch.allclose(m(U), Tuned(Win, W, dtype=torch.float64)(U), atol=1e-12)


@pytest.mark.parametrize("cell", ["gru", "gru_light", "lstm", "mingru"])
def test_gated_trains(data, cell):
    u, y, Win, W = data
    s0, _ = TM.train_regression(u, y, Win, W, 0, kind="gated_tuned", cell=cell, steps=0, dtype=torch.float64)
    s, info = TM.train_regression(u, y, Win, W, 0, kind="gated_tuned", cell=cell, steps=30, dtype=torch.float64)
    assert np.isfinite(s) and s < s0
    n = core.N                                           # g, W_in, biais + par porte : amplitudes (ou gains), entrée, biais
    assert info["n_params"] == {"gru": 9 * n, "gru_light": 9 * n, "lstm": 12 * n, "mingru": 5 * n}[cell]


def test_gated_gates_only_freezes_main_amplitudes(data):
    u, y, Win, W = data
    _, info = TM.train_regression(u, y, Win, W, 0, kind="gated_tuned", cell="gru", gates_only=True, steps=5,
                                  dtype=torch.float64, return_model=True)
    assert torch.all(info["model"].g == 1) and info["n_params"] == 8 * core.N
