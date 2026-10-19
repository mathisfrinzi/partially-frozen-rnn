"""Partie 2.8 et partie 3 : décompositions (Iwasawa, Jordan-Chevalley), quatre formes de couplage, readout carré."""
import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")   # le CI n'installe pas PyTorch : tout le fichier est alors ignoré

from hfrnn import torch_models as TM  # noqa: E402


def _res(n=30, n_in=1, seed=0):
    return TM.make_reservoir(n, n_in=n_in, rho=0.9, in_scale=0.5, sparsity=0.1, seed=seed)


@pytest.mark.parametrize("form", ["iwasawa_A", "iwasawa_N", "iwasawa_AN", "jordan_S", "jordan_SN"])
def test_decomposition_reproduces_reservoir_at_init(form):
    Win, W = _res()
    m = TM.build_model(Win, W, "decomp", 0, torch.float32, "cpu", form=form)
    _, Wm = m.weights()
    assert np.abs(Wm.detach().numpy() - W).max() < 1e-4          # W₀ retrouvée, même pour un réservoir creux


def test_decomposition_parameter_counts():
    Win, W = _res(20)
    count = lambda f: TM.n_trainable(TM.build_model(Win, W, "decomp", 0, torch.float32, "cpu", form=f))   # noqa: E731
    n = 20
    assert count("iwasawa_A") == n + n + n                       # g, entrées, biais
    assert count("iwasawa_N") == n * (n - 1) // 2 + n + n         # S strictement triangulaire, entrées, biais
    assert count("iwasawa_AN") == n + n * (n - 1) // 2 + n + n
    assert count("jordan_SN") == count("jordan_S") + n * (n - 1) // 2


def test_decomposition_trains():
    Win, W = _res()
    u = np.sin(np.arange(1500) * 0.2); y = np.roll(u, 3)
    for form in ("iwasawa_AN", "jordan_SN"):
        err, info = TM.train_regression(u, y, Win, W, 0, kind="decomp", steps=8, form=form)
        assert np.isfinite(err) and info["n_params"] > 0


@pytest.mark.parametrize("form", ["duo1", "duo2", "relay_rand", "relay_learn"])
def test_coupled_forms_shapes_and_training(form):
    Win, W = _res()
    m = TM.build_model(Win, W, "coupled", 0, torch.float32, "cpu", form=form)
    X = m(torch.randn(2, 12, 1))
    assert X.shape == (2, 12, 30) and torch.isfinite(X).all()
    u = np.sin(np.arange(1500) * 0.2); y = np.roll(u, 3)
    err, _ = TM.train_regression(u, y, Win, W, 0, kind="coupled", steps=6, form=form)
    assert np.isfinite(err)


def test_relay_random_keeps_reservoir_block_frozen():
    Win, W = _res()
    m = TM.build_model(Win, W, "coupled", 0, torch.float32, "cpu", form="relay_rand")
    assert not m.WR.requires_grad and not m.WinR.requires_grad
    m2 = TM.build_model(Win, W, "coupled", 0, torch.float32, "cpu", form="relay_rand", learn_R=True)
    assert m2.WR.requires_grad and TM.n_trainable(m2) > TM.n_trainable(m)


def test_squared_readout_doubles_features():
    Win, W = _res()
    m = TM.build_model(Win, W, "spectrum", 0, torch.float32, "cpu", square=True)
    X = m(torch.randn(1, 10, 1))
    assert X.shape[-1] == 60 and torch.allclose(X[..., 30:], X[..., :30] ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        u = np.sin(np.arange(1500) * 0.2); y = np.roll(u, 3)
        TM.train_regression(u, y, Win, W, 0, kind="spectrum", steps=4, square=True)


@pytest.mark.parametrize("a_block,expected", [("free", 30 * 30 + 30 + 30), ("tuned", 30 + 30 + 30),
                                              ("sculpted", 30 * 30 + 30 + 30)])
def test_relay_with_tuned_or_sculpted_block(a_block, expected):
    """Partie 3.4 : le bloc appris de la forme 3 peut être un RNN libre, un réservoir accordé ou un réservoir sculpté."""
    Win, W = _res()
    m = TM.build_model(Win, W, "coupled", 0, torch.float32, "cpu", form="relay_rand", a_block=a_block)
    assert TM.n_trainable(m) == expected
    u = np.sin(np.arange(1500) * 0.2); y = np.roll(u, 3)
    err, _ = TM.train_regression(u, y, Win, W, 0, kind="coupled", steps=4, form="relay_rand", a_block=a_block,
                                 square=True)
    assert np.isfinite(err)


def test_tuned_block_starts_from_its_random_reservoir():
    Win, W = _res()
    m = TM.build_model(Win, W, "coupled", 0, torch.float32, "cpu", form="relay_rand", a_block="tuned")
    _, WA0 = TM.make_reservoir(30, n_in=1, rho=0.9, in_scale=float(np.abs(Win).max()), seed=7000)
    assert np.abs(m.wa().detach().numpy() - WA0).max() < 1e-4


def test_schemas_script_runs(tmp_path, monkeypatch):
    """Les schémas pédagogiques se génèrent sans erreur."""
    import importlib.util
    from pathlib import Path
    monkeypatch.setattr("hfrnn.core.RESULTS_DIR", tmp_path)
    spec = importlib.util.spec_from_file_location(
        "schemas", Path(__file__).resolve().parents[1] / "experiments" / "38_schemas.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.fig_principe(); mod.fig_accorde(); mod.fig_couplage()
    assert (tmp_path / "figures" / "schema_couplage.png").exists()


# ----------------------------------------------------------------------
# Partie 2.9 : accordé tronqué (select = top / bottom / random)
# ----------------------------------------------------------------------
from hfrnn.torch_tuned import Tuned  # noqa: E402


@pytest.mark.parametrize("select,expected", [("top", [0, 1, 2]), ("bottom", [17, 18, 19])])
def test_truncated_tuned_selects_the_right_directions(select, expected):
    Win, W = _res(20)
    m = Tuned(Win, W, dtype=torch.float64, topk=3, select=select)
    assert m.sel.tolist() == expected
    assert TM.n_trainable(m) == 3 + 20 * 1 + 20             # r amplitudes + W_in + biais


def test_truncated_tuned_random_is_reproducible_and_seed_dependent():
    Win, W = _res(30)
    a = Tuned(Win, W, seed=1, dtype=torch.float64, topk=5, select="random").sel.tolist()
    b = Tuned(Win, W, seed=1, dtype=torch.float64, topk=5, select="random").sel.tolist()
    c = Tuned(Win, W, seed=2, dtype=torch.float64, topk=5, select="random").sel.tolist()
    assert a == b and len(set(a)) == 5 and a != c


@pytest.mark.parametrize("select", ["top", "bottom", "random"])
def test_truncated_tuned_starts_at_reservoir_and_moves_only_chosen_gains(select):
    Win, W = _res(20)
    m = Tuned(Win, W, dtype=torch.float64, topk=4, select=select)
    assert np.allclose(m.weights()[1].detach().numpy(), W, atol=1e-9)
    with torch.no_grad():
        m.g += 0.5
    g = m.gains().detach().numpy()
    chosen = set(m.sel.tolist())
    assert all((abs(g[i] - 1.5) < 1e-12) == (i in chosen) for i in range(20))
    assert all(abs(g[i] - 1.0) < 1e-12 for i in range(20) if i not in chosen)


def test_truncated_tuned_rejects_unknown_selection():
    Win, W = _res(20)
    with pytest.raises(ValueError):
        Tuned(Win, W, topk=3, select="middle")


def test_truncated_tuned_gradient_reaches_only_chosen_gains_and_trains():
    Win, W = _res(20)
    m = Tuned(Win, W, dtype=torch.float64, topk=4, select="random")
    out = m(torch.as_tensor(np.random.default_rng(0).normal(size=(2, 15, 1)))).pow(2).sum()
    out.backward()
    assert m.g.grad is not None and m.g.grad.shape == (4,) and torch.isfinite(m.g.grad).all()


# ----------------------------------------------------------------------
# Partie 3.5 : caractéristiques du readout (carré, cube, valeur absolue, produits de paires)
# ----------------------------------------------------------------------
from hfrnn.torch_coupled import FeatureMap  # noqa: E402


@pytest.mark.parametrize("mode", ["square", "cube", "abs", "pairs"])
def test_feature_map_doubles_features_and_keeps_states(mode):
    Win, W = _res(20)
    m = TM.build_model(Win, W, "spectrum", 0, torch.float64, "cpu", feat=mode)
    base = TM.build_model(Win, W, "spectrum", 0, torch.float64, "cpu")
    U = torch.as_tensor(np.random.default_rng(0).normal(size=(2, 10, 1)))
    F, X = m(U), base(U)
    assert m.n_features == 40 and F.shape[-1] == 40
    assert torch.allclose(F[..., :20], X)                      # la première moitié est l'état lui-même
    assert TM.n_trainable(m) == TM.n_trainable(base)           # aucun paramètre récurrent en plus


def test_feature_map_parities_and_pairs():
    Win, W = _res(20)
    base = TM.build_model(Win, W, "reservoir", 0, torch.float64, "cpu")
    X = torch.as_tensor(np.random.default_rng(1).normal(size=(2, 5, 20)))
    for mode, parity in (("square", 1), ("abs", 1), ("cube", -1)):
        f = FeatureMap(base, mode)
        extra = lambda Z: {"square": Z * Z, "abs": Z.abs(), "cube": Z ** 3}[mode]   # noqa: E731
        assert torch.allclose(extra(-X), parity * extra(X))
    a = FeatureMap(base, "pairs", seed=3)
    b = FeatureMap(base, "pairs", seed=3)
    c = FeatureMap(base, "pairs", seed=4)
    assert a.pi.tolist() == b.pi.tolist() and a.pj.tolist() == b.pj.tolist()
    assert (a.pi != a.pj).all() and a.pi.tolist() != c.pi.tolist()
    with pytest.raises(ValueError):
        FeatureMap(base, "tanh")


def test_readout_features_experiment_lists_ten_models():
    import importlib.util
    from pathlib import Path
    spec = importlib.util.spec_from_file_location("rf", Path(__file__).parents[1] / "experiments" / "41_readout_features.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    ms = m.models(100)
    assert len(ms) == 10 and ms[-1][2] == 200 and m.REF in [l for l, _, _ in ms]
    assert sum(1 for _, kw, _ in ms if kw.get("square") or kw.get("feat")) == 6


def test_spectrum_without_g_keeps_recurrence_fixed():
    """learn_g=False : les amplitudes restent à 1, W_in et le biais apprennent (contrôle 4.10)."""
    import numpy as np
    import torch
    from hfrnn import torch_models as TM
    Win, W = TM.make_reservoir(20, n_in=1, rho=0.9, in_scale=0.5, sparsity=0.2, seed=0)
    m = TM.build_model(Win, W, "spectrum", 0, torch.float32, "cpu", learn_g=False)
    assert not m.g.requires_grad and m.Win_l.requires_grad and m.bias.requires_grad
    m2 = TM.build_model(Win, W, "spectrum", 0, torch.float32, "cpu", learn_g=False, learn_win=False)
    assert not m2.g.requires_grad and not m2.Win_l.requires_grad and m2.bias.requires_grad
    _, Wm = m.weights()
    assert np.allclose(Wm.detach().numpy(), W, atol=1e-5)


def test_tuned_without_g_with_rank():
    """Tuned(learn_g=False, rank=r) : g figé, A/B/W_in/biais appris ; compte de paramètres cohérent."""
    import numpy as np
    import torch
    from hfrnn import torch_models as TM
    from hfrnn.torch_tuned import Tuned
    Win, W = TM.make_reservoir(20, n_in=1, rho=0.9, in_scale=0.5, sparsity=0.2, seed=0)
    a = Tuned(Win, W, rank=4)
    b = Tuned(Win, W, rank=4, learn_g=False)
    assert not b.g.requires_grad and a.g.requires_grad
    na = sum(p.numel() for p in a.parameters() if p.requires_grad)
    nb = sum(p.numel() for p in b.parameters() if p.requires_grad)
    assert na - nb == 20
    assert all(p is not b.g for p in b.recurrent_params())


@pytest.mark.parametrize("form", ["iwasawa_N", "jordan_SN"])
def test_decomp_band_limits_parameters_and_keeps_start(form):
    """band=k : seules les k premières sur-diagonales apprennent ; au départ, W = W₀ comme sans bande."""
    from hfrnn.torch_decomp import Decomp
    n, k = 12, 3
    Win, W = _res(n)
    full = Decomp(Win, W, form=form, dtype=torch.float64)
    band = Decomp(Win, W, form=form, dtype=torch.float64, band=k)
    v_full = (full.S if form.startswith("iwasawa") else full.T).numel()
    v_band = (band.S if form.startswith("iwasawa") else band.T).numel()
    assert v_full == n * (n - 1) // 2
    assert v_band == sum(n - d for d in range(1, k + 1))
    assert np.allclose(full.weights()[1].detach().numpy(), band.weights()[1].detach().numpy(), atol=1e-7)
    i, j = band.tri.numpy()
    assert ((j - i) >= 1).all() and ((j - i) <= k).all()
