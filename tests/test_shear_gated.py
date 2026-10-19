"""Réservoir cisaillé à portes (torch_shear.py) : formes, comptes de paramètres, ce qui apprend et ce qui reste figé."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hfrnn import torch_models as TM                      # noqa: E402
from hfrnn.torch_shear import ShearGated                  # noqa: E402

N = 24


def reservoir(seed=0):
    return TM.make_reservoir(N, n_in=1, rho=0.9, in_scale=0.5, sparsity=0.2, seed=seed)


def build(**kw):
    Win, W = reservoir()
    return ShearGated(Win, W, seed=0, dtype=torch.float64, **kw), W


def npar(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def test_candidate_starts_at_reservoir():
    m, W = build(cell="gru", maps=("shear", "shear", "shear"), band=3)
    assert np.allclose(m.cand.matrix().detach().numpy(), W, atol=1e-9)        # Q·R = W₀, S = 0


@pytest.mark.parametrize("cell", ["gru", "mingru", "lin_gate", "linear"])
def test_forward_shape(cell):
    m, _ = build(cell=cell, band=3)
    X = m(torch.randn(2, 30, 1, dtype=torch.float64))
    assert X.shape == (2, 30, N) and torch.isfinite(X).all()


def test_parameter_counts():
    band = 3
    nb = sum(min(band, N - 1 - i) for i in range(N - 1) for _ in [0])          # entrées de la bande, de la forme exacte ci-dessous
    nb = sum(1 for i in range(N) for j in range(i + 1, N) if j - i <= band)
    base = N * 1 + N                                                          # W_in et biais
    shear, _ = build(cell="gru", maps=("shear", "shear", "shear"), band=band)
    assert npar(shear) == base + 3 * nb + 2 * (N + N)                        # trois bandes, B et c des deux portes
    full_gates, _ = build(cell="gru", maps=("full", "full", "shear"), band=band)
    assert npar(full_gates) == base + nb + 2 * N * N + 2 * (N + N)
    light, _ = build(cell="gru", maps=("diag", "diag", "shear"), band=band)
    assert npar(light) == base + nb + 2 * N + 2 * (N + N)
    g, _ = build(cell="mingru", maps=("shear", "shear", "shear"), band=band, learn_g=True)
    assert npar(g) == base + nb + N + (N + N)                                 # une porte (B, c), une bande, g


def test_train_flags_freeze_the_rest():
    for train, want in [(("in",), {"Win_l", "bias", "B_z", "B_r"}),
                        (("gate",), {"c_z", "c_r", "maps.z.S", "maps.r.S"}),
                        (("rec",), {"cand.S"})]:
        m, _ = build(cell="gru", band=2, train=train)
        got = {n for n, p in m.named_parameters() if p.requires_grad}
        assert got == want, (train, got)


def test_gradients_only_where_allowed():
    m, _ = build(cell="gru", band=2, train=("rec",))
    X = m(torch.randn(2, 20, 1, dtype=torch.float64))
    (X ** 2).sum().backward()
    assert m.cand.S.grad is not None and m.cand.S.grad.abs().sum() > 0
    assert m.Win_l.grad is None and m.maps["z"].S.grad is None


def test_trains_through_train_regression():
    u = np.sin(np.arange(400) * 0.2); y = np.roll(u, -3)
    Win, W = reservoir()
    for cell, kw in [("gru", dict(maps=("full", "shear", "shear"))), ("mingru", {}), ("lin_gate", {}), ("linear", {})]:
        s, info = TM.train_regression(u, y, Win, W, 0, kind="shear_gated", steps=5, cell=cell, band=2, **kw)
        assert np.isfinite(s) and info["n_params"] > 0


def test_scattered_shear_has_same_count_as_band():
    from hfrnn.torch_decomp import Decomp
    Win, W = reservoir()
    a = Decomp(Win, W, form="iwasawa_N", band=3, dtype=torch.float64)
    b = Decomp(Win, W, form="iwasawa_N", band=3, scatter=True, dtype=torch.float64)
    assert a.S.numel() == b.S.numel()
    assert not torch.equal(a.tri, b.tri)
    assert (b.tri[1] > b.tri[0]).all()                          # toujours strictement triangulaire supérieur
    X = b(torch.randn(2, 10, 1, dtype=torch.float64))
    assert torch.isfinite(X).all()


@pytest.mark.parametrize("form", ["direct", "direct_tri"])
def test_direct_controls_same_count_and_start_at_reservoir(form):
    from hfrnn.torch_decomp import Decomp
    Win, W = reservoir()
    band = Decomp(Win, W, form="iwasawa_N", band=3, dtype=torch.float64)
    d = Decomp(Win, W, form=form, band=3, dtype=torch.float64)
    assert d.S.numel() == band.S.numel()
    assert np.allclose(d.weights()[1].detach().numpy(), W)                    # S = 0 : exactement le réservoir
    if form == "direct_tri":
        assert (d.tri[1] > d.tri[0]).all()
    else:
        assert (d.tri[1] <= d.tri[0]).any()                                    # des positions hors du triangle
    with torch.no_grad():
        d.S += 0.1
    assert torch.isfinite(d(torch.randn(2, 10, 1, dtype=torch.float64))).all()
