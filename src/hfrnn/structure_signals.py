"""Partie 9 — le gradient au départ dit-il quelle structure apprendre ?

Au réservoir de départ (W₀, W_in, b = 0, lecture ridge ajustée sur ses états), on calcule le gradient de la perte
d'entraînement par fenêtre, G_i = ∂L_i/∂(W, W_in, b), sur des fenêtres tirées comme à l'entraînement. Chaque structure s
(accordé, tourné, cisaillé…) apprend, au premier ordre, un sous-espace T_s de l'espace des matrices W (son espace tangent en
W₀), plus W_in et b. On projette orthogonalement chaque G_i sur T_s ⊕ W_in ⊕ b (métrique de Frobenius) et on en tire des
signaux, comparés ensuite au gain réellement mesuré de chaque structure (parties 3 à 6).

L'espace tangent est obtenu à partir de la jacobienne J_s = ∂vec(W)/∂θ_s en θ_s au départ (autograd sur `weights()` du
modèle, ou formule exacte pour le tourné), puis une base orthonormée U_s de l'image de J_s (SVD). Les coordonnées d'un
gradient dans la structure sont z = (U_sᵀ vec(G_W), vec(G_Win), G_b), et ‖z‖² = ‖P_s(G)‖².
"""
import numpy as np
import torch

from . import core
from . import torch_models as TM

# (nom, réglages) — mêmes noms que results/data/decompositions_n100.json
STRUCTURES = [
    ("RNN", {"kind": "convex", "alpha": 1.0}),
    ("réservoir accordé (valeurs singulières)", {"kind": "spectrum"}),
    ("réservoir tourné (rotations)", {"kind": "rotation"}),
    ("accordé, valeurs propres (Jordan-Chevalley, partie semi-simple)", {"kind": "decomp", "form": "jordan_S"}),
    ("Jordan-Chevalley + partie nilpotente apprise", {"kind": "decomp", "form": "jordan_SN"}),
    ("Iwasawa : amplitudes (A)", {"kind": "decomp", "form": "iwasawa_A"}),
    ("Iwasawa : cisaillement (N)", {"kind": "decomp", "form": "iwasawa_N"}),
    ("Iwasawa : amplitudes et cisaillement (A et N)", {"kind": "decomp", "form": "iwasawa_AN"}),
    ("réservoir cisaillé en bande, k = 1", {"kind": "decomp", "form": "iwasawa_N", "band": 1}),
    ("réservoir cisaillé en bande, k = 3", {"kind": "decomp", "form": "iwasawa_N", "band": 3}),
    ("réservoir cisaillé en bande, k = 10", {"kind": "decomp", "form": "iwasawa_N", "band": 10}),
    ("réservoir nilpotent en bande, k = 3", {"kind": "decomp", "form": "jordan_SN", "band": 3}),
    ("réservoir à biais et entrées appris (sans g)", {"kind": "spectrum", "learn_g": False}),
]
SHORT = {
    "RNN": "RNN", "réservoir accordé (valeurs singulières)": "accordé", "réservoir tourné (rotations)": "tourné",
    "accordé, valeurs propres (Jordan-Chevalley, partie semi-simple)": "valeurs propres",
    "Jordan-Chevalley + partie nilpotente apprise": "nilpotent", "Iwasawa : amplitudes (A)": "Iwasawa A",
    "Iwasawa : cisaillement (N)": "cisaillé plein", "Iwasawa : amplitudes et cisaillement (A et N)": "cisaillé A et N",
    "réservoir cisaillé en bande, k = 1": "bande k=1", "réservoir cisaillé en bande, k = 3": "bande k=3",
    "réservoir cisaillé en bande, k = 10": "bande k=10", "réservoir nilpotent en bande, k = 3": "nilpotent k=3",
    "réservoir à biais et entrées appris (sans g)": "biais et entrées",
}


def reservoir(n, d, seed):
    return TM.make_reservoir(n, n_in=d, rho=core.RHO, in_scale=core.IN_SCALE / np.sqrt(d), sparsity=core.SPARSITY,
                             seed=seed)


# --------------------------------------------------------------------------------------------- gradients au départ
def ridge_readout(u, y, nt, Win, W, b=None):
    """Lecture ridge du réservoir (biais b, nuls par défaut), ajustée sur la partie entraînement : (w, c)."""
    with torch.no_grad():
        ut = torch.as_tensor(u, dtype=torch.float64)
        b = torch.zeros(W.shape[0], dtype=torch.float64) if b is None else torch.as_tensor(b, dtype=torch.float64)
        X = _run(ut.view(1, -1, 1) if ut.dim() == 1 else ut.unsqueeze(0), torch.as_tensor(Win), torch.as_tensor(W),
                 b)[0]
        wc = TM.ridge(X[core.WASHOUT:nt], torch.as_tensor(y[core.WASHOUT:nt]))
    return wc[:-1], wc[-1]


def _run(U, Win, W, b):
    x = torch.zeros(U.shape[0], W.shape[0], dtype=U.dtype)
    xs = []
    for t in range(U.shape[1]):
        x = torch.tanh(U[:, t] @ Win.T + x @ W.T + b)
        xs.append(x)
    return torch.stack(xs, 1)


def window_grads(u, y, starts, Win, W, w, c, b=None, chunk=256):
    """Gradient de la perte de chaque fenêtre (même perte qu'à l'entraînement) par rapport à W, W_in et b, au point
    (Win, W, b) (b nul par défaut, c'est le départ).
    Renvoie trois tableaux (B, N²), (B, N·d), (B, N)."""
    from torch.func import grad, vmap
    ut = torch.as_tensor(u, dtype=torch.float64)
    yt = torch.as_tensor(y, dtype=torch.float64)
    idx = torch.as_tensor(np.asarray(starts)[:, None] + np.arange(core.WIN_LEN)[None, :])
    U = ut[idx] if ut.dim() == 2 else ut[idx].unsqueeze(-1)
    Y = yt[idx]
    Wt, Wint = torch.as_tensor(W, dtype=torch.float64), torch.as_tensor(Win, dtype=torch.float64)
    b0 = torch.zeros(W.shape[0], dtype=torch.float64) if b is None else torch.as_tensor(b, dtype=torch.float64)
    w, c = torch.as_tensor(w, dtype=torch.float64), torch.as_tensor(c, dtype=torch.float64)

    def loss(Wm, Wi, b, Ui, Yi):
        X = _run(Ui.unsqueeze(0), Wi, Wm, b)[0]
        pred = X @ w + c
        return ((pred[core.LOSS_WASH:] - Yi[core.LOSS_WASH:]) ** 2).mean()

    f = vmap(grad(loss, argnums=(0, 1, 2)), in_dims=(None, None, None, 0, 0))
    outs = [f(Wt, Wint, b0, U[i:i + chunk], Y[i:i + chunk]) for i in range(0, U.shape[0], chunk)]
    gW, gI, gb = (torch.cat([o[k] for o in outs]) for k in range(3))
    B = U.shape[0]
    return gW.reshape(B, -1).numpy(), gI.reshape(B, -1).numpy(), gb.numpy()


# --------------------------------------------------------------------------------------------- espaces tangents
def tangent_jacobian(Win, W, seed, kw):
    """J = ∂vec(W)/∂θ en θ de départ (N² × p) pour la partie récurrente de la structure ; "full" pour le RNN, None
    pour une structure sans paramètre récurrent (biais et entrées seuls)."""
    kw = dict(kw)
    kind = kw.pop("kind")
    n = W.shape[0]
    if kind == "convex":
        return "full"
    if kind == "rotation":                         # W(Ω) = e^Ω W₀ e^{−Ω} : dW = Ω W₀ − W₀ Ω, Ω antisymétrique
        iu = np.triu_indices(n, 1)
        cols = np.empty((n * n, len(iu[0])))
        for k, (i, j) in enumerate(zip(*iu)):
            Om = np.zeros((n, n)); Om[i, j], Om[j, i] = 1.0, -1.0
            cols[:, k] = (Om @ W - W @ Om).ravel()
        return cols
    model = TM.build_model(Win, W, kind, seed, torch.float64, "cpu", **kw)
    params = {k: p.detach().clone() for k, p in model.named_parameters()
              if p.requires_grad and not k.endswith("Win_l") and not k.endswith("bias")}
    if not params:
        return None

    class Wrap(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self):
            return self.m.weights()[1].reshape(-1)

    wrap = Wrap(model)
    from torch.func import functional_call, jacrev
    f = lambda ps: functional_call(wrap, {"m." + k: v for k, v in ps.items()}, ())   # noqa: E731
    jac = jacrev(f, chunk_size=256)(params)                  # par morceaux : la mémoire reste bornée
    return torch.cat([jac[k].reshape(n * n, -1) for k in params], 1).numpy()


def orthonormal_basis(J, rtol=1e-8):
    """Base orthonormée de l'image de J (SVD)."""
    if isinstance(J, str) or J is None:
        return J
    Uj, s, _ = np.linalg.svd(J, full_matrices=False)
    return np.ascontiguousarray(Uj[:, s > rtol * s.max()], dtype=np.float32)   # float32 : quatre bases de 10⁴ × 5·10³


def coords(gW, gI, gb, U):
    """Coordonnées des gradients dans T_s ⊕ W_in ⊕ b, dans une base orthonormée : ‖z‖ = ‖P_s(G)‖."""
    if isinstance(U, str):
        zW = gW
    elif U is None:
        zW = np.zeros((gW.shape[0], 0))
    else:
        zW = gW @ U
    return np.concatenate([zW, gI, gb], 1)


# --------------------------------------------------------------------------------------------- signaux
def signals(Ztr, Zva, Gtr_full, lam, pcs):
    """Signaux d'une structure à partir des coordonnées de ses gradients par fenêtre (entraînement, validation).
    Gtr_full : gradients complets (B, D), pour les parts relatives ; lam : régularisation du pas de Newton ;
    pcs : premières directions principales des gradients complets (r, D), déjà projetées (r, dim) pour cette structure."""
    B, dim = Ztr.shape
    gbar, gval = Ztr.mean(0), Zva.mean(0)
    full = (Gtr_full.mean(0) ** 2).sum()
    cap = float((gbar ** 2).sum() / full)
    noise = float(((Ztr - gbar) ** 2).sum(1).mean() / B)            # variance de la moyenne
    snr = float((gbar ** 2).sum() / max(noise, 1e-300))
    cos = float(gbar @ gval / (np.linalg.norm(gbar) * np.linalg.norm(gval) + 1e-300))
    # pas de Newton régularisé dans la structure : ḡᵀ (F + λI)⁻¹ ḡ avec F = ZᵀZ/B (Fisher empirique), par Woodbury :
    # (λI + ZᵀZ/B)⁻¹ = (I − Zᵀ (λB·I + Z Zᵀ)⁻¹ Z) / λ
    a = Ztr @ gbar
    newton = float((gbar @ gbar - a @ np.linalg.solve(lam * B * np.eye(B) + Ztr @ Ztr.T, a)) / lam)
    pca = float((pcs ** 2).sum(1).mean())
    return {"part captée": cap, "part par paramètre": cap / dim, "signal sur bruit": snr,
            "accord entraînement-validation": cos, "pas de Newton": newton, "directions principales": pca,
            "dimension": dim}


def probe(u, y, nt, Win, W, b, seed, structures, B=128, r_pcs=10, bases=None):
    """Signaux du gradient de chaque structure au point (Win, W, b) : {structure: signaux}. Même protocole qu'au départ
    (experiments/44_structure_signals.py), mais valable en n'importe quel point de l'entraînement."""
    w, c = ridge_readout(u, y, nt, Win, W, b)
    nv = int(0.75 * nt)                                         # entraînement / validation, comme en 8.2
    rng = np.random.default_rng(500 + seed)
    tr = rng.integers(0, nv - core.WIN_LEN - 1, B)
    va = rng.integers(nv, nt - core.WIN_LEN - 1, B)
    gtr, gva = window_grads(u, y, tr, Win, W, w, c, b), window_grads(u, y, va, Win, W, w, c, b)
    Gtr = np.concatenate(gtr, 1)
    lam = float((Gtr ** 2).sum(1).mean() / Gtr.shape[1])
    _, _, Vt = np.linalg.svd(Gtr, full_matrices=False)
    pcs = Vt[:r_pcs]
    nW, nI = W.shape[0] ** 2, Win.size
    out = {}
    for s, kw in structures:
        base = bases[s] if bases is not None else orthonormal_basis(tangent_jacobian(Win, W, seed, kw))
        Ztr, Zva = coords(*gtr, base), coords(*gva, base)
        P = coords(pcs[:, :nW], pcs[:, nW:nW + nI], pcs[:, nW + nI:], base)
        out[s] = signals(Ztr, Zva, Gtr, lam, P)
    return out
