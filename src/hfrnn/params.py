"""Paramétrisations « réservoir aléatoire + partie apprise » et entraîneur commun.

Paramétrisations : Convex (mélange α, apprentissage ralenti), LearnedAlpha, LowRank, SparseMask
(partie 1) ; SuperMask (partie 2) ; Rotation, SpectrumOnly (partie 3).

Chaque paramétrisation dit :
  - quels tableaux sont appris (`params`) et avec quel multiplicateur de lr (`lr_mult`),
  - comment construire les poids effectifs (`compose`) à partir du réservoir fixe,
  - comment transformer dLoss/dW_in et dLoss/dW (sortis du BPTT) en gradients
    de ses propres paramètres (`grads`, règle de la chaîne).

`train(...)` fait le reste : BPTT + Adam sur des fenêtres tirées avec une graine fixe,
gradient clippé à 1 (norme globale), readout de départ nul, readout final recalculé
par ridge, NMSE sur le test.
"""
import numpy as np

from . import core
from .core import run_states, ridge_fit, ridge_predict, nmse, spectral_radius, bptt, Adam


class Convex:
    """W = (1 - alpha) W₀ + alpha ΔW, ΔW initialisé à W₀ (idem pour W_in).

    alpha = 1 : RNN classique.  lr_mult = f : W_in et W apprennent f fois moins vite
    (lr_mult_in permet de donner un autre multiplicateur à W_in).
    """

    def __init__(self, alpha=1.0, lr_mult=1.0, lr_mult_in=None):
        self.alpha, self.mult = float(alpha), float(lr_mult)
        self.mult_in = self.mult if lr_mult_in is None else float(lr_mult_in)

    def init(self, Win_r, W_r, rng):
        self.Win_l, self.W_l = Win_r.copy(), W_r.copy()
        self.params, self.lr_mult = [self.Win_l, self.W_l], [self.mult_in, self.mult]

    def compose(self, Win_r, W_r):
        a = self.alpha
        return (1 - a) * Win_r + a * self.Win_l, (1 - a) * W_r + a * self.W_l

    def grads(self, dWin, dW, Win_r, W_r):
        return [self.alpha * dWin, self.alpha * dW]

    def info(self, Win_r, W_r):
        return {}


class LearnedAlpha:
    """Comme Convex, mais alpha = sigmoid(theta) est appris : un alpha pour W_in, un pour W.

    dLoss/dalpha = somme(G ⊙ (ΔW - W₀)). theta a son propre lr (lr × lr_mult_theta)
    et n'entre pas dans le clipping global.
    """

    def __init__(self, alpha0=0.1, lr_mult_theta=5.0):
        self.alpha0, self.mult = float(alpha0), float(lr_mult_theta)

    def init(self, Win_r, W_r, rng):
        self.theta = np.full(2, np.log(self.alpha0 / (1 - self.alpha0)))
        self.Win_l, self.W_l = Win_r.copy(), W_r.copy()
        self.params, self.lr_mult = [self.theta, self.Win_l, self.W_l], [self.mult, 1.0, 1.0]
        self.noclip = {0}

    def alphas(self):
        return 1 / (1 + np.exp(-self.theta))

    def compose(self, Win_r, W_r):
        a = self.alphas()
        return (1 - a[0]) * Win_r + a[0] * self.Win_l, (1 - a[1]) * W_r + a[1] * self.W_l

    def grads(self, dWin, dW, Win_r, W_r):
        a = self.alphas()
        dalpha = np.array([(dWin * (self.Win_l - Win_r)).sum(), (dW * (self.W_l - W_r)).sum()])
        return [dalpha * a * (1 - a), a[0] * dWin, a[1] * dW]

    def info(self, Win_r, W_r):
        a = self.alphas()
        return {"alpha_in": float(a[0]), "alpha_rec": float(a[1])}


class LowRank:
    """W = W₀ + U V^T (rang r, style LoRA : U aléatoire, V = 0 au départ).

    W_in = W_in₀ + ΔW_in (vecteur de taille N, appris en entier, peu coûteux).
    rank = 0 : seul W_in et le biais sont appris, la dynamique récurrente reste figée.
    """

    def __init__(self, rank, u_scale=None):
        self.r = int(rank)
        self.u_scale = u_scale

    def init(self, Win_r, W_r, rng):
        n = W_r.shape[0]
        scale = self.u_scale if self.u_scale is not None else 1 / np.sqrt(n)
        self.U = rng.normal(0, scale, (n, self.r))
        self.V = np.zeros((n, self.r))
        self.Win_l = np.zeros_like(Win_r)
        self.params, self.lr_mult = [self.Win_l, self.U, self.V], [1.0, 1.0, 1.0]

    def compose(self, Win_r, W_r):
        return Win_r + self.Win_l, W_r + self.U @ self.V.T

    def grads(self, dWin, dW, Win_r, W_r):
        return [dWin, dW @ self.V, dW.T @ self.U]

    def info(self, Win_r, W_r):
        return {"rel": float(np.linalg.norm(self.U @ self.V.T) / np.linalg.norm(W_r))}


class SparseMask:
    """W = W₀ + M ⊙ ΔW, M = connexions existantes du réservoir (10 %).

    On n'ajuste que les poids déjà présents : même graphe que le réservoir.
    """

    def init(self, Win_r, W_r, rng):
        self.M = (W_r != 0).astype(float)
        self.Win_l, self.W_l = np.zeros_like(Win_r), np.zeros_like(W_r)
        self.params, self.lr_mult = [self.Win_l, self.W_l], [1.0, 1.0]

    def compose(self, Win_r, W_r):
        return Win_r + self.Win_l, W_r + self.M * self.W_l

    def grads(self, dWin, dW, Win_r, W_r):
        return [dWin, self.M * dW]

    def info(self, Win_r, W_r):
        return {"rel": float(np.linalg.norm(self.M * self.W_l) / np.linalg.norm(W_r))}


def train(param, u, y, Win_r, W_r, seed, steps=None, lr=None, log_every=0, init="zero"):
    """Entraîne une paramétrisation ; renvoie (NMSE test, infos).

    init : readout de départ. "zero" (défaut), "ridge" (ridge sur les états de départ,
    lambda = 1e-6) ou un nombre (ridge avec ce lambda). Avec "ridge", les poids du readout
    de départ peuvent être énormes et déstabiliser le début de l'entraînement : à éviter.
    """
    steps = core.STEPS if steps is None else steps
    lr = core.LR if lr is None else lr
    n = W_r.shape[0]
    rng = np.random.default_rng(1000 + seed)            # mêmes fenêtres que models.run_model
    NT = core.n_train(u)
    param.init(Win_r, W_r, np.random.default_rng(2000 + seed))
    bias = np.zeros(n)

    Win_t, W_t = param.compose(Win_r, W_r)
    if init == "zero":
        w0 = np.zeros(n + 1)
    else:
        lam = core.RIDGE if init == "ridge" else float(init)
        X0 = run_states(u[:NT], Win_t, W_t, bias)
        w0 = ridge_fit(X0[core.WASHOUT:], y[core.WASHOUT:NT], lam=lam)
    Wout, c = w0[:n].reshape(n, 1).copy(), w0[n:].copy()
    info0 = {"wout0_norm": float(np.linalg.norm(Wout))}

    tensors = param.params + [bias, Wout, c]
    opt = Adam(tensors, [lr * m for m in param.lr_mult] + [lr] * 3)
    losses = []
    for step in range(steps):
        Win_t, W_t = param.compose(Win_r, W_r)
        starts = rng.integers(0, NT - core.WIN_LEN - 1, core.BATCH)
        idx = starts[:, None] + np.arange(core.WIN_LEN)[None, :]
        loss, dWin, dW, db, dWout, dc = bptt(u[idx], y[idx], Win_t, W_t, bias, Wout, c)
        grads = param.grads(dWin, dW, Win_r, W_r) + [db, dWout, dc]
        noclip = getattr(param, "noclip", set())
        gn = np.sqrt(sum((g ** 2).sum() for i, g in enumerate(grads) if i not in noclip))
        if gn > 1.0:
            grads = [g if i in noclip else g / gn for i, g in enumerate(grads)]
        opt.step(grads)
        if log_every and step % log_every == 0:
            losses.append(float(loss))

    Win_t, W_t = param.compose(Win_r, W_r)
    X = run_states(u, Win_t, W_t, bias)
    w = ridge_fit(X[core.WASHOUT:NT], y[core.WASHOUT:NT])
    score = nmse(ridge_predict(X, w)[NT:], y[NT:])
    info = dict(param.info(Win_r, W_r), rho=float(spectral_radius(W_t)), **info0)
    # variantes d'évaluation propres à une paramétrisation (ex. masque binarisé après entraînement)
    for name, (Win_v, W_v) in getattr(param, "eval_variants", lambda a, b: {})(Win_r, W_r).items():
        Xv = run_states(u, Win_v, W_v, bias)
        wv = ridge_fit(Xv[core.WASHOUT:NT], y[core.WASHOUT:NT])
        info[name] = float(nmse(ridge_predict(Xv, wv)[NT:], y[NT:]))
    if log_every:
        info["loss"] = losses
    return float(score), info


def pure_reservoir(u, y, Win_r, W_r):
    """Réservoir classique : aucun entraînement, readout ridge."""
    NT = core.n_train(u)
    X = run_states(u, Win_r, W_r, np.zeros(W_r.shape[0]))
    w = ridge_fit(X[core.WASHOUT:NT], y[core.WASHOUT:NT])
    return float(nmse(ridge_predict(X, w)[NT:], y[NT:]))


# ----------------------------------------------------------------------
# Partie 2 : supermasque — on ne change jamais les poids aléatoires, on choisit lesquels garder
# ----------------------------------------------------------------------
class SuperMask:
    """W = m ⊙ W₀ : les valeurs aléatoires ne changent jamais, on apprend un masque m.

    mode "sigmoid" : m = sigmoid(s) continu dans [0, 1], gradient exact, départ s0 (m ≈ 1 : tout est gardé).
    mode "topk"    : m binaire, on garde la fraction `keep` des connexions existantes ayant les plus hauts
                     scores ; gradient « straight-through » (comme edge-popup, Ramanujan et al., 2020).
    mode "gain"    : contrôle, m = s libre (départ 1) : gain par connexion sans borne, qui peut changer
                     le signe ou amplifier. Équivaut à un RNN dont chaque poids apprend à une vitesse
                     proportionnelle à sa valeur aléatoire.
    rescale        : multiplie W par 1/sqrt(keep) (mode topk) pour compenser la perte d'amplitude.
    win            : "fixed" (W_in aléatoire figé), "free" (W_in = W_in₀ + ΔW_in),
                     "mask" (même masque appliqué à W_in, mode sigmoid uniquement).
    """

    def __init__(self, mode="sigmoid", keep=0.5, win="free", lr_mult=10.0, s0=3.0, rescale=False):
        self.mode, self.keep, self.win = mode, float(keep), win
        self.mult, self.s0, self.rescale = float(lr_mult), float(s0), bool(rescale)

    def init(self, Win_r, W_r, rng):
        self.support = W_r != 0
        if self.mode == "sigmoid":
            self.s = np.full(W_r.shape, self.s0)
        elif self.mode == "gain":
            self.s = np.ones(W_r.shape)
        else:                                          # scores aléatoires (comme edge-popup)
            self.s = rng.uniform(0, 1, W_r.shape)
        self.params, self.lr_mult = [self.s], [self.mult]
        if self.win == "free":
            self.Win_l = np.zeros_like(Win_r)
            self.params.append(self.Win_l); self.lr_mult.append(1.0)
        elif self.win == "mask":
            assert self.mode in ("sigmoid", "gain")
            self.s_in = np.full(Win_r.shape, self.s0) if self.mode == "sigmoid" else np.ones(Win_r.shape)
            self.params.append(self.s_in); self.lr_mult.append(self.mult)
        self.c = 1 / np.sqrt(self.keep) if (self.rescale and self.mode == "topk") else 1.0

    def mask(self):
        if self.mode == "sigmoid":
            return 1 / (1 + np.exp(-self.s))
        if self.mode == "gain":
            return self.s * self.support
        vals = self.s[self.support]
        k = max(1, int(round(self.keep * vals.size)))
        thr = np.partition(vals, vals.size - k)[vals.size - k]
        return ((self.s >= thr) & self.support).astype(float)

    def compose(self, Win_r, W_r):
        W = self.c * self.mask() * W_r
        if self.win == "free":
            return Win_r + self.Win_l, W
        if self.win == "mask":
            m_in = self.s_in if self.mode == "gain" else 1 / (1 + np.exp(-self.s_in))
            return m_in * Win_r, W
        return Win_r, W

    def grads(self, dWin, dW, Win_r, W_r):
        g = self.c * dW * W_r
        if self.mode == "sigmoid":
            m = self.mask()
            g = g * m * (1 - m)
        out = [g]
        if self.win == "free":
            out.append(dWin)
        elif self.win == "mask":
            if self.mode == "gain":
                out.append(dWin * Win_r)
            else:
                mi = 1 / (1 + np.exp(-self.s_in))
                out.append(dWin * Win_r * mi * (1 - mi))
        return out

    def eval_variants(self, Win_r, W_r):
        """Masque continu binarisé après entraînement (garder si m > 0.5) : est-ce un simple élagage ?"""
        if self.mode != "sigmoid":
            return {}
        Win_t, _ = self.compose(Win_r, W_r)
        if self.win == "mask":
            Win_t = (self.s_in > 0).astype(float) * Win_r
        return {"nmse_binarise": (Win_t, (self.s > 0).astype(float) * W_r)}

    def info(self, Win_r, W_r):
        m = self.mask()[self.support]
        return {"kept": float((m > 0.5).mean()), "mask_mean": float(m.mean()),
                "rho_mask": float(spectral_radius(self.compose(Win_r, W_r)[1]))}


# ----------------------------------------------------------------------
# Partie 3 : spectre contre directions
# ----------------------------------------------------------------------
class Rotation:
    """W = Q · W₀ · Qᵀ avec Q = exp(S − Sᵀ) orthogonale : valeurs propres et singulières de W₀
    conservées exactement, on apprend seulement l'orientation de la dynamique.

    win : "free" (W_in = W_in₀ + ΔW_in), "rotate" (W_in = Q · W_in₀), "fixed".
    Avec tanh, une rotation n'est pas une reparamétrisation (elle le serait pour un réseau linéaire).
    """

    def __init__(self, win="free", lr_mult=1.0):
        self.win, self.mult = win, float(lr_mult)

    def init(self, Win_r, W_r, rng):
        self.S = np.zeros_like(W_r)
        self.params, self.lr_mult = [self.S], [self.mult]
        if self.win == "free":
            self.Win_l = np.zeros_like(Win_r)
            self.params.append(self.Win_l); self.lr_mult.append(1.0)

    def Q(self):
        from scipy.linalg import expm
        return expm(self.S - self.S.T)

    def compose(self, Win_r, W_r):
        Q = self._Q = self.Q()
        W = Q @ W_r @ Q.T
        if self.win == "free":
            return Win_r + self.Win_l, W
        if self.win == "rotate":
            return Q @ Win_r, W
        return Win_r, W

    def grads(self, dWin, dW, Win_r, W_r):
        from scipy.linalg import expm_frechet
        A = self.S - self.S.T
        Q = self.Q()
        dQ = dW @ Q @ W_r.T + dW.T @ Q @ W_r
        if self.win == "rotate":
            dQ = dQ + dWin @ Win_r.T
        dA = expm_frechet(A.T, dQ, compute_expm=False)        # adjoint de la dérivée de expm
        out = [dA - dA.T]
        if self.win == "free":
            out.append(dWin)
        return out

    def info(self, Win_r, W_r):
        Q = self.Q()
        return {"rot_dist": float(np.linalg.norm(Q - np.eye(len(Q))) / np.sqrt(len(Q)))}


class SpectrumOnly:
    """W = U · diag(σ ⊙ g) · Vᵀ (décomposition en valeurs singulières de W₀) : directions figées,
    on apprend seulement l'amplification g de chaque direction (N paramètres). W_in libre."""

    def __init__(self, lr_mult=1.0):
        self.mult = float(lr_mult)

    def init(self, Win_r, W_r, rng):
        self.U, self.sig, Vt = np.linalg.svd(W_r)
        self.V = Vt.T
        self.g = np.ones_like(self.sig)
        self.Win_l = np.zeros_like(Win_r)
        self.params, self.lr_mult = [self.g, self.Win_l], [self.mult, 1.0]

    def compose(self, Win_r, W_r):
        return Win_r + self.Win_l, (self.U * (self.sig * self.g)) @ self.V.T

    def grads(self, dWin, dW, Win_r, W_r):
        return [self.sig * np.einsum("ij,ik,kj->j", self.U, dW, self.V), dWin]

    def info(self, Win_r, W_r):
        return {"g_min": float(self.g.min()), "g_max": float(self.g.max())}


# ----------------------------------------------------------------------
# Partie 4 : combinaisons
# ----------------------------------------------------------------------
class Combo:
    """Combinaisons des parties 1 à 3 :

        B = U · diag(σ ⊙ g) · Vᵀ   si spectrum (réservoir accordé), sinon B = W₀
        W = m ⊙ B                   si mask (réservoir sculpté, m = sigmoid(s) ∈ [0, 1])
          + ΔW                      si slow (correction libre apprise lentement, lr × slow : « freiné »)
        W_in = W_in₀ + ΔW_in (win = "free") ou m' ⊙ W_in₀ (win = "mask", rien d'appris sur les poids)

    Le masque porte sur toute la matrice B (avec le spectre, B est dense même si W₀ est creuse).
    Départ : g = 1, m = sigmoid(s0) ≈ 0.95, ΔW = 0.
    """

    def __init__(self, spectrum=False, mask=False, slow=None, win="free", mask_mult=50.0, spec_mult=1.0, s0=3.0):
        assert spectrum or mask or slow, "au moins une composante"
        self.spectrum, self.use_mask, self.slow, self.win = spectrum, mask, slow, win
        self.mask_mult, self.spec_mult, self.s0 = float(mask_mult), float(spec_mult), float(s0)

    def init(self, Win_r, W_r, rng):
        self.params, self.lr_mult = [], []
        if self.spectrum:
            self.U, self.sig, Vt = np.linalg.svd(W_r)
            self.V = Vt.T
            self.g = np.ones_like(self.sig)
            self.params.append(self.g); self.lr_mult.append(self.spec_mult)
        if self.use_mask:
            self.s = np.full(W_r.shape, self.s0)
            self.params.append(self.s); self.lr_mult.append(self.mask_mult)
        if self.slow:
            self.D = np.zeros_like(W_r)
            self.params.append(self.D); self.lr_mult.append(float(self.slow))
        if self.win == "free":
            self.Win_l = np.zeros_like(Win_r)
            self.params.append(self.Win_l); self.lr_mult.append(1.0)
        else:
            self.s_in = np.full(Win_r.shape, self.s0)
            self.params.append(self.s_in); self.lr_mult.append(self.mask_mult)

    def base(self, W_r):
        return (self.U * (self.sig * self.g)) @ self.V.T if self.spectrum else W_r

    def mask(self):
        return 1 / (1 + np.exp(-self.s))

    def compose(self, Win_r, W_r):
        B = self.base(W_r)
        W = self.mask() * B if self.use_mask else B.copy()
        if self.slow:
            W = W + self.D
        if self.win == "free":
            return Win_r + self.Win_l, W
        return Win_r / (1 + np.exp(-self.s_in)), W

    def grads(self, dWin, dW, Win_r, W_r):
        out = []
        dB = dW * self.mask() if self.use_mask else dW
        if self.spectrum:
            out.append(self.sig * np.einsum("ij,ik,kj->j", self.U, dB, self.V))
        if self.use_mask:
            m = self.mask()
            out.append(dW * self.base(W_r) * m * (1 - m))
        if self.slow:
            out.append(dW)
        if self.win == "free":
            out.append(dWin)
        else:
            mi = 1 / (1 + np.exp(-self.s_in))
            out.append(dWin * Win_r * mi * (1 - mi))
        return out

    def info(self, Win_r, W_r):
        d = {"rho": float(spectral_radius(self.compose(Win_r, W_r)[1]))}
        if self.spectrum:
            d.update(g_min=float(self.g.min()), g_max=float(self.g.max()))
        if self.use_mask:
            d["mask_mean"] = float(self.mask().mean())
        if self.slow:
            d["dW_rel"] = float(np.linalg.norm(self.D) / np.linalg.norm(W_r))
        return d
