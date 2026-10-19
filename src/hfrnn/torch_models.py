"""Version PyTorch (GPU) des modèles « réservoir + partie apprise ».

Même protocole que la version numpy (hfrnn.params.train) pour la régression :
fenêtres tirées avec la même graine, Adam, gradient clippé à 1 (norme globale),
readout de départ nul, readout final en ridge. En float64 sur CPU, `train_regression` reproduit
la version numpy (vérifié par tests/test_torch.py) ; sur GPU on passe en float32
pour la vitesse.

Paramétrisations disponibles (argument `kind`) :
    "reservoir"  rien d'appris dans la partie récurrente (réservoir classique, readout seul)
    "convex"     W = (1 - alpha) W₀ + alpha ΔW   (alpha fixe ; lr_mult, lr_mult_in possibles)
    "lalpha"     idem avec alpha appris : 2 scalaires (alpha_in, alpha_rec)
    "lowrank"    W = W₀ + U V^T, W_in = W_in₀ + ΔW_in
    "sparse"     W = W₀ + M ⊙ ΔW (M = connexions du réservoir)
    "supermask"  W = m ⊙ W₀, masque appris (partie 2)
    "rotation"   W = Q W₀ Qᵀ, Q orthogonale apprise (partie 3)
    "spectrum"   W = U diag(σ ⊙ g) Vᵀ, amplifications g apprises (partie 3)
    "combo"      W = m ⊙ [U diag(σ ⊙ g) Vᵀ ou W₀] + ΔW lent (partie 4, comme params.Combo)
    "shear_gated"  réservoir cisaillé à portes (partie 5, voir torch_shear.py)
    "gated_tuned"  réservoir accordé à portes, type GRU ou LSTM (partie 5, voir torch_tuned.py)
    "linear_tuned" réservoir accordé linéaire, type LRU : seules les valeurs propres et les entrées apprennent
                   (partie 5, voir torch_baselines.py)
    "tuned", "deep_tuned"  améliorations du réservoir accordé (partie 4.2, voir torch_tuned.py)
    "gru", "lstm", "lru"  modèles existants pour comparaison (partie 5, voir torch_baselines.py)
"""
import math

import numpy as np
import torch

from . import core


def make_reservoir(n, n_in=1, rho=0.9, in_scale=0.5, sparsity=0.1, seed=0):
    """Réservoir numpy (même générateur que core.make_reservoir quand n_in = 1)."""
    rng = np.random.default_rng(seed)
    Win = rng.uniform(-in_scale, in_scale, (n, n_in))
    W = rng.uniform(-1, 1, (n, n)) * (rng.random((n, n)) < sparsity)
    W *= rho / max(abs(np.linalg.eigvals(W)))
    return Win, W


class MixRNN(torch.nn.Module):
    def __init__(self, Win_r, W_r, kind="convex", alpha=1.0, lr_mult=1.0, rank=0, seed=0,
                 alpha0=0.1, train_bias=True, dtype=torch.float32, device="cpu", lr_mult_in=None,
                 mode="sigmoid", keep=0.5, win="free", mult=None, s0=3.0, rescale=False,
                 spectrum=False, mask=False, slow=None, mask_mult=50.0, spec_mult=1.0, learn_win=True, learn_g=True):
        super().__init__()
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)   # noqa: E731
        self.register_buffer("Win_r", f(Win_r))
        self.register_buffer("W_r", f(W_r))
        n = W_r.shape[0]
        self.kind, self.lr_mult = kind, float(lr_mult)
        self.lr_mult_in = self.lr_mult if lr_mult_in is None else float(lr_mult_in)
        P = torch.nn.Parameter
        if kind == "convex":
            self.alpha = float(alpha)
            self.Win_l, self.W_l = P(f(Win_r).clone()), P(f(W_r).clone())
        elif kind == "lalpha":
            self.Win_l, self.W_l = P(f(Win_r).clone()), P(f(W_r).clone())
            self.theta = P(torch.full((2,), math.log(alpha0 / (1 - alpha0)), dtype=dtype, device=device))
        elif kind == "lowrank":
            rng = np.random.default_rng(2000 + seed)            # comme params.LowRank
            self.U = P(f(rng.normal(0, 1 / np.sqrt(n), (n, rank))))
            self.V = P(torch.zeros(n, rank, dtype=dtype, device=device))
            self.Win_l = P(torch.zeros_like(f(Win_r)))
        elif kind == "sparse":
            self.register_buffer("M", (f(W_r) != 0).to(dtype))
            self.Win_l = P(torch.zeros_like(f(Win_r)))
            self.W_l = P(torch.zeros_like(f(W_r)))
        elif kind == "supermask":                           # comme params.SuperMask
            self.mode, self.keep, self.win = mode, float(keep), win
            self.mult = 10.0 if mult is None else float(mult)
            self.register_buffer("support", f(W_r) != 0)
            if mode == "sigmoid":
                self.s = P(torch.full_like(f(W_r), s0))
            else:
                self.s = P(f(np.random.default_rng(2000 + seed).uniform(0, 1, W_r.shape)))
            if win == "free":
                self.Win_l = P(torch.zeros_like(f(Win_r)))
            elif win == "mask":
                self.s_in = P(torch.full_like(f(Win_r), s0))
            self.c = 1 / math.sqrt(self.keep) if (rescale and mode == "topk") else 1.0
        elif kind == "rotation":                            # comme params.Rotation
            self.win, self.mult = win, (1.0 if mult is None else float(mult))
            self.S = P(torch.zeros_like(f(W_r)))
            if win == "free":
                self.Win_l = P(torch.zeros_like(f(Win_r)))
        elif kind == "spectrum":                            # comme params.SpectrumOnly
            self.mult = 1.0 if mult is None else float(mult)
            U_, sig, Vt = np.linalg.svd(np.asarray(W_r))
            self.register_buffer("Us", f(U_)); self.register_buffer("sig", f(sig)); self.register_buffer("Vs", f(Vt.T))
            self.g = P(torch.ones_like(f(sig)))
            self.Win_l = P(torch.zeros_like(f(Win_r)))
        elif kind == "combo":                               # comme params.Combo
            self.spectrum, self.use_mask, self.slow, self.win = spectrum, mask, slow, win
            self.mask_mult, self.spec_mult = float(mask_mult), float(spec_mult)
            if spectrum:
                U_, sig, Vt = np.linalg.svd(np.asarray(W_r))
                self.register_buffer("Us", f(U_)); self.register_buffer("sig", f(sig))
                self.register_buffer("Vs", f(Vt.T))
                self.g = P(torch.ones_like(f(sig)))
            if mask:
                self.s = P(torch.full_like(f(W_r), s0))
            if slow:
                self.D = P(torch.zeros_like(f(W_r)))
            if win == "free":
                self.Win_l = P(torch.zeros_like(f(Win_r)))
            else:
                self.s_in = P(torch.full_like(f(Win_r), s0))
        elif kind != "reservoir":
            raise ValueError(kind)
        self.bias = P(torch.zeros(n, dtype=dtype, device=device), requires_grad=train_bias and kind != "reservoir")
        if not learn_win and hasattr(self, "Win_l"):           # entrées figées (partie 7.3 : plusieurs tâches)
            self.Win_l.requires_grad_(False)
        if not learn_g and hasattr(self, "g"):                 # amplitudes figées à 1 : contrôle « W_in et biais seuls » (4.10)
            self.g.requires_grad_(False)

    # poids effectifs
    def weights(self):
        if self.kind == "convex":
            a = self.alpha
            return (1 - a) * self.Win_r + a * self.Win_l, (1 - a) * self.W_r + a * self.W_l
        if self.kind == "lalpha":
            a = torch.sigmoid(self.theta)
            return (1 - a[0]) * self.Win_r + a[0] * self.Win_l, (1 - a[1]) * self.W_r + a[1] * self.W_l
        if self.kind == "lowrank":
            return self.Win_r + self.Win_l, self.W_r + self.U @ self.V.T
        if self.kind == "sparse":
            return self.Win_r + self.Win_l, self.W_r + self.M * self.W_l
        if self.kind == "supermask":
            if self.mode == "sigmoid":
                m = torch.sigmoid(self.s)
            else:                                           # top-k + gradient straight-through
                vals = self.s[self.support]
                k = max(1, int(round(self.keep * vals.numel())))
                thr = torch.kthvalue(vals.detach(), vals.numel() - k + 1).values
                hard = ((self.s >= thr) & self.support).to(self.s.dtype)
                m = hard + self.s - self.s.detach()
            W = self.c * m * self.W_r
            if self.win == "free":
                return self.Win_r + self.Win_l, W
            if self.win == "mask":
                return torch.sigmoid(self.s_in) * self.Win_r, W
            return self.Win_r, W
        if self.kind == "rotation":
            Q = torch.linalg.matrix_exp(self.S - self.S.T)
            W = Q @ self.W_r @ Q.T
            if self.win == "free":
                return self.Win_r + self.Win_l, W
            if self.win == "rotate":
                return Q @ self.Win_r, W
            return self.Win_r, W
        if self.kind == "spectrum":
            return self.Win_r + self.Win_l, (self.Us * (self.sig * self.g)) @ self.Vs.T
        if self.kind == "combo":
            W = (self.Us * (self.sig * self.g)) @ self.Vs.T if self.spectrum else self.W_r
            if self.use_mask:
                W = torch.sigmoid(self.s) * W
            if self.slow:
                W = W + self.D
            Win = self.Win_r + self.Win_l if self.win == "free" else torch.sigmoid(self.s_in) * self.Win_r
            return Win, W
        return self.Win_r, self.W_r

    def recurrent_params(self):
        """Paramètres de la partie récurrente (hors alpha appris)."""
        return [p for name, p in self.named_parameters()
                if name in ("Win_l", "W_l", "U", "V", "s", "s_in", "S", "g", "D")]

    def param_groups(self, lr):
        """Groupes Adam : W_in appris à lr × lr_mult_in, le reste de la partie récurrente à lr × lr_mult."""
        named = dict(self.named_parameters())
        rec = [named[k] for k in ("W_l", "U", "V") if k in named]
        inp = [named[k] for k in ("Win_l",) if k in named]
        special = [named[k] for k in ("s", "s_in", "S", "g") if k in named]   # masque, rotation, spectre
        groups = []
        if self.kind == "combo":                            # un groupe par composante, comme params.Combo
            for k, m in (("g", self.spec_mult if self.kind == "combo" else 1), ("s", self.mask_mult),
                         ("D", self.slow or 1.0), ("Win_l", 1.0), ("s_in", self.mask_mult)):
                if k in named:
                    groups.append({"params": [named[k]], "lr": lr * m})
            return groups
        if rec:
            groups.append({"params": rec, "lr": lr * self.lr_mult})
        if inp:
            groups.append({"params": inp, "lr": lr * self.lr_mult_in})
        if special:
            groups.append({"params": special, "lr": lr * self.mult})
        return groups

    def forward(self, U):
        """U : (B, L, n_in) -> états (B, L, n), état initial nul."""
        Win, W = self.weights()
        B, L, _ = U.shape
        x = torch.zeros(B, W.shape[0], dtype=U.dtype, device=U.device)
        xs = []
        for t in range(L):
            x = torch.tanh(U[:, t] @ Win.T + x @ W.T + self.bias)
            xs.append(x)
        return torch.stack(xs, 1)


def ridge(X, y, lam=None):
    """Régression ridge, toujours calculée en float64 (lam = 1e-6 est trop petit pour du float32).

    Forme duale quand il y a plus de caractéristiques que d'échantillons (très grands réservoirs) :
    w = Aᵀ (A Aᵀ + λ I)⁻¹ y, identique à la forme primale."""
    lam = core.RIDGE if lam is None else lam
    X, y = X.double(), y.double()
    A = torch.cat([X, torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)], 1)
    if A.shape[1] > A.shape[0]:
        return A.T @ torch.linalg.solve(A @ A.T + lam * torch.eye(A.shape[0], dtype=X.dtype, device=X.device), y)
    return torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1], dtype=X.dtype, device=X.device), A.T @ y)


def build_model(Win_r, W_r, kind, seed, dtype, device, **kw):
    """Nos modèles (MixRNN) ou un modèle existant (GRU, LSTM, LRU) de même taille d'état."""
    if kw.pop("square", False):                              # readout carré (contrôle NVAR, partie 3)
        from .torch_coupled import SquaredFeatures
        return SquaredFeatures(build_model(Win_r, W_r, kind, seed, dtype, device, **kw))
    feat = kw.pop("feat", None)                              # partie 3.5 : square / cube / abs / pairs
    if feat:
        from .torch_coupled import FeatureMap
        return FeatureMap(build_model(Win_r, W_r, kind, seed, dtype, device, **kw), feat, seed)
    if kind == "tuned":
        from .torch_tuned import Tuned
        return Tuned(Win_r, W_r, seed=seed, dtype=dtype, device=device, **kw)
    if kind in ("fast_tuned", "fast_reservoir"):
        from .torch_tuned import FastTuned
        return FastTuned(Win_r, seed=seed, dtype=dtype, device=device, learn=(kind == "fast_tuned"), **kw)
    if kind == "gated_tuned":
        from .torch_tuned import GatedTuned
        return GatedTuned(Win_r, W_r, seed=seed, dtype=dtype, device=device, **kw)
    if kind == "shear_gated":                                # réservoir cisaillé à portes (partie 5)
        from .torch_shear import ShearGated
        return ShearGated(Win_r, W_r, seed=seed, dtype=dtype, device=device, **kw)
    if kind == "subspace":                                   # partie 9 : sous-espace choisi par le gradient
        from .torch_subspace import Subspace
        return Subspace(Win_r, W_r, seed=seed, dtype=dtype, device=device, **kw)
    if kind == "deep_tuned":
        from .torch_tuned import DeepTuned
        return DeepTuned(W_r.shape[0], n_in=Win_r.shape[1], seed=seed, dtype=dtype, device=device, **kw)
    if kind == "decomp":                                     # partie 2 : Iwasawa, Jordan-Chevalley
        from .torch_decomp import Decomp
        return Decomp(Win_r, W_r, seed=seed, dtype=dtype, device=device, **kw)
    if kind == "coupled":                                    # partie 5 bis
        from .torch_coupled import Coupled
        return Coupled(Win_r, W_r, seed=seed, dtype=dtype, device=device, **kw)
    if kind in ("gru", "lstm", "lru"):
        from .torch_baselines import Baseline
        return Baseline(kind, n=W_r.shape[0], n_in=Win_r.shape[1], seed=seed, dtype=dtype, device=device, **kw)
    if kind == "linear_tuned":                               # réservoir accordé linéaire, type LRU (partie 5)
        from .torch_baselines import Baseline
        return Baseline("lru", n=W_r.shape[0], n_in=Win_r.shape[1], seed=seed, dtype=dtype, device=device,
                        tuned=True, **kw)
    return MixRNN(Win_r, W_r, kind=kind, seed=seed, dtype=dtype, device=device, **kw)


def n_trainable(model):
    """Nombre de paramètres appris dans la partie récurrente (hors readout)."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


@torch.no_grad()
def as_batch(ut):
    """Série (T,) ou (T, d) -> lot (1, T, d) : une entrée scalaire ou un vecteur d'entrées par pas."""
    return ut.view(1, -1, 1) if ut.dim() == 1 else ut.unsqueeze(0)


@torch.no_grad()
def regression_nmse(model, ut, yt, n_train=None):
    """Readout ridge sur les états de la série complète, NMSE sur la partie test (après n_train pas)."""
    NT, WO = (core.n_train(ut) if n_train is None else n_train), core.WASHOUT
    X = model(as_batch(ut))[0]
    w = ridge(X[WO:NT], yt[WO:NT])
    Xd, yd = X.double(), yt.double()
    pred = torch.cat([Xd, torch.ones(len(Xd), 1, dtype=Xd.dtype, device=Xd.device)], 1) @ w
    return float(((pred[NT:] - yd[NT:]) ** 2).mean() / yd[NT:].var(unbiased=False))


def train_regression(u, y, Win_r, W_r, seed, kind="convex", steps=None, lr=None, device="cpu",
                     dtype=torch.float32, lr_theta=0.01, init="zero", eval_at=(), return_model=False, model=None,
                     n_train=None, **kw):
    """Équivalent PyTorch de params.train. Renvoie (NMSE, infos).

    init : readout de départ, "zero" (défaut, comme la version numpy), "ridge" ou un lambda.
    eval_at : pas d'entraînement où mesurer aussi la NMSE (infos["checkpoints"] = {pas: NMSE}).
    model : modèle déjà construit (et éventuellement déjà entraîné) à entraîner de nouveau, au lieu d'en construire
            un neuf : sert à apprendre plusieurs tâches l'une après l'autre (partie 7.3).
    u : série (T,) ou (T, d) (plusieurs entrées par pas, partie 8.5).
    n_train : nombre de pas d'entraînement (défaut : 75 %, soit 3 000 sur 4 000) ; le reste sert au test.
    """
    steps = core.STEPS if steps is None else steps
    lr = core.LR if lr is None else lr
    rng = np.random.default_rng(1000 + seed)
    model = build_model(Win_r, W_r, kind, seed, dtype, device, **kw) if model is None else model
    n = model.n_features if hasattr(model, "n_features") else W_r.shape[0]
    ut = torch.as_tensor(u, dtype=dtype, device=device)
    yt = torch.as_tensor(y, dtype=dtype, device=device)
    NT, WO = (core.n_train(u) if n_train is None else n_train), core.WASHOUT

    with torch.no_grad():                                  # readout initial
        if init == "zero":
            w0 = torch.zeros(n + 1, dtype=torch.float64, device=device)
        else:
            X0 = model(as_batch(ut[:NT]))[0]
            w0 = ridge(X0[WO:], yt[WO:NT], lam=None if init == "ridge" else float(init))
    Wout = torch.nn.Parameter(w0[:n].clone().view(n, 1).to(dtype))
    c = torch.nn.Parameter(w0[n:].clone().to(dtype))

    main = model.recurrent_params()
    others = ([model.bias] if model.bias.requires_grad else []) + [Wout, c]
    groups = model.param_groups(lr) + [{"params": others, "lr": lr}]
    opt = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8)
    opt_theta = torch.optim.Adam([model.theta], lr=lr_theta) if kind == "lalpha" else None
    clip_params = main + others
    if kind in ("reservoir", "fast_reservoir"):
        steps = 0                                          # réservoir classique : aucun entraînement
    checkpoints = {}

    for step in range(steps):
        starts = rng.integers(0, NT - core.WIN_LEN - 1, core.BATCH)
        idx = torch.as_tensor(starts[:, None] + np.arange(core.WIN_LEN)[None, :], device=device)
        U = ut[idx] if ut.dim() == 2 else ut[idx].unsqueeze(-1)
        X = model(U)                                       # (B, L, n)
        pred = (X @ Wout)[..., 0] + c
        loss = ((pred[:, core.LOSS_WASH:] - yt[idx][:, core.LOSS_WASH:]) ** 2).mean()
        if hasattr(model, "penalty"):                      # ex. rester près du réservoir (partie 4.2)
            loss = loss + model.penalty()
        opt.zero_grad()
        if opt_theta is not None:
            opt_theta.zero_grad()
        loss.backward()
        # même règle que numpy : division par la norme globale si > 1
        gn = torch.sqrt(sum((p.grad ** 2).sum() for p in clip_params if p.grad is not None))
        if gn > 1.0:
            for p in clip_params:
                if p.grad is not None:
                    p.grad /= gn
        opt.step()
        if opt_theta is not None:
            opt_theta.step()
        if step + 1 in eval_at and step + 1 < steps:
            checkpoints[step + 1] = regression_nmse(model, ut, yt, NT)

    err = regression_nmse(model, ut, yt, NT)
    checkpoints[steps] = err
    info = {"checkpoints": checkpoints, "n_params": n_trainable(model)}
    if return_model:
        info["model"] = model
    if kind == "lalpha":
        a = torch.sigmoid(model.theta).detach().cpu().numpy()
        info.update(alpha_in=float(a[0]), alpha_rec=float(a[1]))
    return float(err), info


# ----------------------------------------------------------------------
# Classification de séquences (MNIST ligne par ligne : 28 pas de 28 pixels)
# ----------------------------------------------------------------------
class SeqClassifier(torch.nn.Module):
    def __init__(self, core_rnn, n_classes=10):
        super().__init__()
        self.rnn = core_rnn
        n = getattr(core_rnn, "n_features", None) or core_rnn.W_r.shape[0]
        ref = core_rnn.W_r if hasattr(core_rnn, "W_r") else next(core_rnn.parameters())
        self.readout = torch.nn.Linear(2 * n, n_classes).to(ref.device, ref.dtype)

    def forward(self, U):
        X = self.rnn(U)
        feats = torch.cat([X[:, -1], X.mean(1)], 1)        # dernier état + état moyen
        return self.readout(feats)


def train_classifier(train_loader, test_loader, Win_r, W_r, seed, kind="convex", epochs=3, lr=1e-3,
                     device="cuda", log=print, lr_theta=0.01, **kw):
    """Entraîne un SeqClassifier ; renvoie (précision test, historique)."""
    torch.manual_seed(seed)
    rnn = MixRNN(Win_r, W_r, kind=kind, seed=seed, device=device, **kw)
    model = SeqClassifier(rnn)
    main = rnn.recurrent_params()
    others = ([rnn.bias] if rnn.bias.requires_grad else []) + list(model.readout.parameters())
    groups = [{"params": others, "lr": lr}] + rnn.param_groups(lr)
    opt = torch.optim.Adam(groups)
    opt_theta = torch.optim.Adam([rnn.theta], lr=lr_theta) if kind == "lalpha" else None
    hist = []
    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device).view(-1, 28, 28), yb.to(device)
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            opt.zero_grad()
            if opt_theta is not None:
                opt_theta.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(main + others, 1.0)
            opt.step()
            if opt_theta is not None:
                opt_theta.step()
        acc = evaluate(model, test_loader, device)
        hist.append(acc)
        log(f"    {kind} {kw} seed {seed} époque {ep + 1}: précision test {acc:.4f}")
    return hist[-1], hist


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ok = tot = 0
    for xb, yb in loader:
        xb, yb = xb.to(device).view(-1, 28, 28), yb.to(device)
        ok += (model(xb).argmax(1) == yb).sum().item()
        tot += len(yb)
    return ok / tot


@torch.no_grad()
def accuracy(model, X, y, batch=None):
    """Précision d'un classifieur sur des tenseurs (N, L, n_in) et (N,) déjà sur l'appareil."""
    batch = batch or (2000 if X.shape[1] <= 100 else 500)     # longues séquences : lots plus petits (mémoire)
    model.eval()
    ok = sum(int((model(X[i:i + batch]).argmax(1) == y[i:i + batch]).sum()) for i in range(0, len(y), batch))
    return ok / len(y)


def fit_classifier(data, kind, seed, n=256, rho=0.9, in_scale=0.1, density=0.1, epochs=3, lr=1e-3, batch=128,
                   device="cpu", test=False, **kw):
    """Classifieur de séquences, données déjà sur l'appareil : (N, L, n_in), ex. 28 pas de 28 pixels (MNIST ligne par
    ligne) ou 784 pas d'un pixel (MNIST pixel par pixel).

    data : dict Xtr, ytr, Xva, yva (et Xte, yte si test=True). `kind` : tout ce que build_model accepte (MixRNN,
    tuned, gru, lstm...). Le test n'est mesuré (à chaque époque) que si test=True : la recherche d'hyperparamètres
    n'utilise que la validation. Un entraînement qui diverge (perte non finie) s'arrête et compte comme le hasard
    (1 / nombre de classes, déduit des étiquettes 0 … K−1).
    Renvoie {"val": [...], "test": [...] ou None, "n_params": int, "diverged": bool}.
    """
    import time
    t0 = time.time()
    torch.manual_seed(seed)
    n_in = data["Xtr"].shape[-1]
    n_classes = int(max(int(data["ytr"].max()), int(data["yva"].max()))) + 1
    if kind in ("gru", "lstm", "lru", "linear_tuned"):       # pas de réservoir dense : seules les tailles comptent
        Win, W = np.zeros((n, n_in)), np.zeros((n, n))
    else:
        Win, W = make_reservoir(n, n_in=n_in, rho=rho, in_scale=in_scale, sparsity=density, seed=seed)
    rnn = build_model(Win, W, kind, seed, torch.float32, device, **kw)
    model = SeqClassifier(rnn, n_classes)
    others = ([rnn.bias] if rnn.bias.requires_grad else []) + list(model.readout.parameters())
    opt = torch.optim.Adam(rnn.param_groups(lr) + [{"params": others, "lr": lr}])
    clip = rnn.recurrent_params() + others
    Xtr, ytr = data["Xtr"], data["ytr"]
    gen = torch.Generator().manual_seed(seed)
    val, tst, diverged = [], [], False
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(ytr), generator=gen).to(device)
        for i in range(0, len(perm) - batch + 1, batch):
            idx = perm[i:i + batch]
            loss = torch.nn.functional.cross_entropy(model(Xtr[idx]), ytr[idx])
            if not torch.isfinite(loss):
                diverged = True
                break
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clip, 1.0)
            opt.step()
        if diverged:
            chance = 1 / n_classes                              # une précision de hasard
            val += [chance] * (epochs - ep); tst += [chance] * (epochs - ep) if test else []
            break
        val.append(accuracy(model, data["Xva"], data["yva"]))
        if test:
            tst.append(accuracy(model, data["Xte"], data["yte"]))
    return {"val": val, "test": tst if test else None, "n_params": n_trainable(rnn), "diverged": diverged,
            "seconds": time.time() - t0}
