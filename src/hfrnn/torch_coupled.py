"""Partie 3 — réservoir et RNN en couplage : quatre formes (et leurs combinaisons avec l'accordé et le sculpté).

R = bloc réservoir (poids aléatoires figés), A = bloc appris. Deux états x et y de N neurones. Le readout lit
un seul état (N caractéristiques, comme les autres modèles).

    duo1   duo croisé, lu côté RNN        y' = tanh(WinR u + W_R y + C x)   (R, figé sauf C)
                                          x' = tanh(WinA u + W_A x + D y)   (A, appris)   readout sur x
    duo2   duo croisé, lu côté réservoir  mêmes équations, readout sur y
    relay_rand   relais aléatoire         y(t) = tanh(WinR u(t) + W_R x(t))       figé, sans mémoire
                                          x(t+1) = tanh(WinA u(t) + W_A y(t))     appris
    relay_learn  relais appris            y(t) = tanh(WinA u(t) + W_A x(t))       appris, sans mémoire
                                          x(t+1) = tanh(WinR u(t) + W_R y(t))     figé    readout sur x

Appris : W_A, W_inA (et C, D pour les duos). Figé : W_R, W_inR. Le biais est appris.

a_block (combinaisons, partie 3.4) : la forme du bloc appris W_A.
    "free"      W_A appris en entier (défaut, N² paramètres)
    "tuned"     W_A = U·diag(σ ⊙ g)·Vᵀ, décomposition d'un réservoir aléatoire : seules les amplitudes g apprennent
                (réservoir accordé dans la transition), et les entrées W_inA (départ aléatoire + correction apprise)
    "sculpted"  W_A = m ⊙ W_A0, m = sigmoid(s) ∈ [0, 1] (réservoir sculpté dans la transition ; lr des scores × 50)
"""
import numpy as np
import torch

from .torch_models import make_reservoir


class Coupled(torch.nn.Module):
    def __init__(self, Win_r, W_r, form="relay_rand", seed=0, dtype=torch.float32, device="cpu", couple=0.3,
                 train_bias=True, learn_R=False, a_block="free", mask_mult=50.0):
        super().__init__()
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)   # noqa: E731
        n, d = W_r.shape[0], Win_r.shape[1]
        self.form, self.n_features = form, n
        if learn_R:                                          # contrôle : bloc R aussi appris (transition profonde, DT-RNN)
            self.WinR, self.WR = torch.nn.Parameter(f(Win_r)), torch.nn.Parameter(f(W_r))
        else:
            self.register_buffer("WinR", f(Win_r)); self.register_buffer("WR", f(W_r))
        # bloc appris : tirage indépendant du bloc réservoir, de même loi (puis il apprend librement)
        WinA, WA = make_reservoir(n, n_in=d, rho=0.9, in_scale=float(np.abs(Win_r).max()), seed=7000 + seed)
        P = torch.nn.Parameter
        self.a_block, self.mask_mult = a_block, float(mask_mult)
        if a_block == "free":
            self.WinA_free, self.WA_free = P(f(WinA)), P(f(WA))
        elif a_block in ("tuned", "sculpted"):
            self.register_buffer("WinA0", f(WinA))
            self.WinA_l = P(torch.zeros_like(f(WinA)))
            if a_block == "tuned":
                U_, sig, Vt = np.linalg.svd(WA)
                self.register_buffer("Us", f(U_)); self.register_buffer("sig", f(sig)); self.register_buffer("Vs", f(Vt.T))
                self.g = P(torch.ones(n, dtype=dtype, device=device))
            else:
                self.register_buffer("WA0", f(WA))
                self.s = P(torch.full((n, n), 3.0, dtype=dtype, device=device))
        else:
            raise ValueError(a_block)
        if form in ("duo1", "duo2"):
            _, C = make_reservoir(n, rho=couple, sparsity=0.1, seed=8000 + seed)
            _, D = make_reservoir(n, rho=couple, sparsity=0.1, seed=9000 + seed)
            self.C, self.D = P(f(C)), P(f(D))
        elif form not in ("relay_rand", "relay_learn"):
            raise ValueError(form)
        self.bias = P(torch.zeros(n, dtype=dtype, device=device), requires_grad=train_bias)

    def wa(self):
        if self.a_block == "free":
            return self.WA_free
        if self.a_block == "tuned":
            return (self.Us * (self.sig * self.g)) @ self.Vs.T
        return torch.sigmoid(self.s) * self.WA0

    def wina(self):
        return self.WinA_free if self.a_block == "free" else self.WinA0 + self.WinA_l

    def recurrent_params(self):
        return [p for k, p in self.named_parameters() if k != "bias" and p.requires_grad]

    def param_groups(self, lr):
        if self.a_block == "sculpted":
            rest = [p for k, p in self.named_parameters() if k not in ("bias", "s") and p.requires_grad]
            return [{"params": rest, "lr": lr}, {"params": [self.s], "lr": lr * self.mask_mult}]
        return [{"params": self.recurrent_params(), "lr": lr}]

    def forward(self, U):
        B, L, _ = U.shape
        n = self.n_features
        x = torch.zeros(B, n, dtype=U.dtype, device=U.device)
        y = torch.zeros_like(x)
        out = []
        f = self.form
        WA, WinA = self.wa(), self.wina()
        for t in range(L):
            u = U[:, t]
            if f in ("duo1", "duo2"):
                y, x = (torch.tanh(u @ self.WinR.T + y @ self.WR.T + x @ self.C.T),
                        torch.tanh(u @ WinA.T + x @ WA.T + y @ self.D.T + self.bias))
                out.append(x if f == "duo1" else y)
            elif f == "relay_rand":
                y = torch.tanh(u @ self.WinR.T + x @ self.WR.T)
                x = torch.tanh(u @ WinA.T + y @ WA.T + self.bias)
                out.append(x)
            else:                                                 # relay_learn
                y = torch.tanh(u @ WinA.T + x @ WA.T + self.bias)
                x = torch.tanh(u @ self.WinR.T + y @ self.WR.T)
                out.append(x)
        return torch.stack(out, 1)


class SquaredFeatures(torch.nn.Module):
    """Contrôle NVAR / readout carré : le readout lit [x, x²] au lieu de x (2 N caractéristiques).
    Enveloppe n'importe quel modèle ; les paramètres appris sont ceux du modèle enveloppé."""

    def __init__(self, inner):
        super().__init__()
        nf = getattr(inner, "n_features", None) or inner.W_r.shape[0]
        self.inner, self.n_features = inner, 2 * nf

    @property
    def bias(self):
        return self.inner.bias

    def recurrent_params(self):
        return self.inner.recurrent_params()

    def param_groups(self, lr):
        return self.inner.param_groups(lr)

    def forward(self, U):
        X = self.inner(U)
        return torch.cat([X, X * X], -1)


class FeatureMap(SquaredFeatures):
    """Contrôles du readout carré (partie 3.5) : le readout lit [x, φ(x)] au lieu de x, avec 2 N caractéristiques dans
    tous les cas, pour que seule la *nature* des caractéristiques ajoutées change.

        mode="square"  φ(x) = x²            fonction paire, ordre 2        (= SquaredFeatures)
        mode="cube"    φ(x) = x³            fonction IMPAIRE de même ordre de grandeur : si la symétrie impaire du tanh
                                            est la cause, elle n'aide pas
        mode="abs"     φ(x) = |x|           paire, non polynomiale
        mode="pairs"   φ(x)_k = x_i · x_j   produits de N paires d'états tirées au hasard (i ≠ j, graine propre à `seed`)
    """

    MODES = ("square", "cube", "abs", "pairs")

    def __init__(self, inner, mode="square", seed=0):
        super().__init__(inner)
        if mode not in self.MODES:
            raise ValueError(mode)
        self.mode = mode
        if mode == "pairs":
            n = self.n_features // 2
            rng = np.random.default_rng(9000 + seed)
            i = rng.integers(0, n, n)
            j = (i + rng.integers(1, n, n)) % n                  # j ≠ i
            self.register_buffer("pi", torch.as_tensor(i))
            self.register_buffer("pj", torch.as_tensor(j))

    def forward(self, U):
        X = self.inner(U)
        if self.mode == "square":
            extra = X * X
        elif self.mode == "cube":
            extra = X * X * X
        elif self.mode == "abs":
            extra = X.abs()
        else:
            extra = X[..., self.pi] * X[..., self.pj]
        return torch.cat([X, extra], -1)
