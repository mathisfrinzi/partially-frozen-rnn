"""Partie 4.2 — améliorations du réservoir accordé (version PyTorch).

Réservoir accordé (partie 3) :   W = U · diag(σ ⊙ g) · Vᵀ    (W₀ = U·diag(σ)·Vᵀ, g appris, U et V figés)

Variantes (combinables) de `Tuned` :
    gmax=c      accordé borné : g = c · sigmoid(s) ∈ [0, c]   (la leçon de la partie 2 : borner aide)
    leak=True   vitesse propre par neurone : x(t+1) = (1 − a) ⊙ x(t) + a ⊙ tanh(…), a = sigmoid(c) appris
                (départ a ≈ 0.98, presque le réservoir d'origine)
    basis="eig" accordé sur les valeurs propres : W = Re(P · diag(λ ⊙ g ⊙ e^{iφ}) · P⁻¹), module g et phase φ
                appris pour chaque mode (mêmes valeurs pour deux modes conjugués, donc W reste réel)
    poly=K      accordé paramétrique : g = 1 + Σₖ cₖ·Tₖ(z), polynômes de Tchebychev de la position z ∈ [−1, 1]
                de chaque direction dans le spectre trié ; K coefficients au lieu de N
    rank=r      + correction de rang r : W = … + A·Bᵀ (B nul au départ), comme le réservoir corrigé
    selective   accordé sélectif : les amplitudes dépendent de l'entrée, g(t) = g + G·u(t) (G nul au départ)
    neuron_gains  gains par neurone des deux côtés : W = diag(d_L)·U·diag(σ ⊙ g)·Vᵀ·diag(d_R)
    penalty=λ   rester près du réservoir : pénalité λ · moyenne((g − 1)²) ajoutée à la perte
    rotations=K quelques rotations apprises entre K paires de directions (angles nuls au départ)

Contrôles de la partie 3 (pourquoi l'accordé marche) :
    spectrum="flat"    même directions aléatoires, mais toutes les valeurs singulières égales (réservoir orthogonal)
    spectrum="spread"  même directions, valeurs singulières très étalées (géométriques sur 3 ordres de grandeur)
                       (dans les deux cas, rayon spectral ramené à 0.9 comme le réservoir aléatoire)
    topk=k             seules k directions sont accordées, les autres gardent g = 1 ; lesquelles : select="top" (les k
                       plus grandes valeurs singulières, défaut), "bottom" (les k plus petites) ou "random" (k tirées
                       au hasard, graine propre à `seed`) ; la partie 2.9 compare les trois

`FastTuned` (partie 7) : accordé à directions structurées et rapides, W = P₁ · F⁻¹ · diag(λ ⊙ g) · F · P₂
(F : transformée de Fourier réelle, P₁, P₂ : permutations aléatoires fixes). Coût N·log N par pas au lieu de N².

`DeepTuned` : deux réservoirs accordés de N/2 neurones empilés (le second lit les états du premier) ;
caractéristiques = les deux états concaténés (N au total, comme les autres modèles).
"""
import math

import numpy as np
import torch

from .core import IN_SCALE, RHO


def _spectral_radius(W):
    return float(np.max(np.abs(np.linalg.eigvals(W))))


class Tuned(torch.nn.Module):
    def __init__(self, Win_r, W_r, seed=0, dtype=torch.float32, device="cpu",
                 basis="svd", gmax=None, leak=False, poly=None, rank=0, selective=False, neuron_gains=False,
                 penalty=0.0, rotations=0, spectrum="random", topk=None, select="top", learn_g=True):
        super().__init__()
        if select not in ("top", "bottom", "random"):
            raise ValueError(select)
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)   # noqa: E731
        P = torch.nn.Parameter
        n = W_r.shape[0]
        self.n_features = n
        self.basis, self.gmax, self.use_leak, self.poly, self.rank = basis, gmax, leak, poly, rank
        self.selective, self.neuron_gains, self.pen, self.rotations = selective, neuron_gains, penalty, rotations
        self.topk = topk
        if basis != "svd" and (selective or neuron_gains or rotations or spectrum != "random"):
            raise ValueError("ces options demandent basis='svd'")
        self.register_buffer("Win_r", f(Win_r))
        self.register_buffer("W_r", f(W_r))
        W0 = np.asarray(W_r, dtype=float)
        if basis == "svd":
            U_, sig, Vt = np.linalg.svd(W0)
            if spectrum != "random":                            # contrôle : autre spectre, mêmes directions
                sig = np.full(n, 1.0) if spectrum == "flat" else np.geomspace(1.0, 1e-3, n)
                sig = sig * RHO / _spectral_radius((U_ * sig) @ Vt)
            self.register_buffer("Us", f(U_)); self.register_buffer("sig", f(sig)); self.register_buffer("Vs", f(Vt.T))
            order = np.arange(n)                                # déjà trié par σ décroissante
        elif basis == "eig":
            lam, Pm = np.linalg.eig(W0)
            cdt = torch.complex64 if dtype == torch.float32 else torch.complex128
            c = lambda a: torch.as_tensor(a, dtype=cdt, device=device)   # noqa: E731
            self.register_buffer("Pm", c(Pm)); self.register_buffer("Pinv", c(np.linalg.pinv(Pm)))   # pinv : robuste si W₀ est presque défective
            self.register_buffer("lam", c(lam))
            pair = np.array([int(np.argmin(np.abs(lam - np.conj(l)))) for l in lam])
            self.register_buffer("pair", torch.as_tensor(pair, device=device))
            self.phi = P(torch.zeros(n, dtype=dtype, device=device))
            order = np.argsort(-np.abs(lam), kind="stable")
        else:
            raise ValueError(basis)
        if poly:
            z = np.empty(n); z[order] = np.linspace(-1, 1, n)  # position de chaque direction dans le spectre
            T = np.polynomial.chebyshev.chebvander(z, poly)[:, 1:]            # T₁ … T_K
            self.register_buffer("T", f(T))
            self.coef = P(torch.zeros(poly, dtype=dtype, device=device))
        elif gmax:
            self.s = P(torch.full((n,), math.log(1 / (gmax - 1)) if gmax != 2 else 0.0, dtype=dtype, device=device))
        elif topk:
            if select == "top":                                 # les k plus grandes valeurs singulières
                sel = np.arange(topk)
            elif select == "bottom":                            # contrôle : les k plus petites
                sel = np.arange(n - topk, n)
            else:                                               # contrôle : k directions tirées au hasard
                sel = np.sort(np.random.default_rng(5000 + seed).permutation(n)[:topk])
            self.register_buffer("sel", torch.as_tensor(sel, device=device))
            self.g = P(torch.ones(topk, dtype=dtype, device=device))
        else:
            self.g = P(torch.ones(n, dtype=dtype, device=device))
        if selective:
            self.G = P(torch.zeros(n, Win_r.shape[1], dtype=dtype, device=device))
        if neuron_gains:
            self.dL = P(torch.ones(n, dtype=dtype, device=device))
            self.dR = P(torch.ones(n, dtype=dtype, device=device))
        if rotations:
            perm = np.random.default_rng(4000 + seed).permutation(n)[:2 * rotations]
            self.register_buffer("rot_i", torch.as_tensor(perm[0::2], device=device))
            self.register_buffer("rot_j", torch.as_tensor(perm[1::2], device=device))
            self.theta = P(torch.zeros(rotations, dtype=dtype, device=device))
        if leak:
            self.c_leak = P(torch.full((n,), 4.0, dtype=dtype, device=device))
        if rank:
            rng = np.random.default_rng(2000 + seed)
            self.A = P(f(rng.normal(0, 1 / np.sqrt(n), (n, rank))))
            self.B = P(torch.zeros(n, rank, dtype=dtype, device=device))
        self.Win_l = P(torch.zeros_like(f(Win_r)))
        self.bias = P(torch.zeros(n, dtype=dtype, device=device))
        if not learn_g:                                         # contrôle « sans g » (4.10) : amplitudes figées à 1
            if not hasattr(self, "g"):
                raise ValueError("learn_g=False suppose les amplitudes g libres (ni gmax, ni poly)")
            self.g.requires_grad_(False)

    def gains(self):
        if self.poly:
            return 1 + self.T @ self.coef
        if self.gmax:
            return self.gmax * torch.sigmoid(self.s)
        if self.topk:                                           # g = 1 partout, sauf sur les directions choisies
            return torch.ones(self.n_features, dtype=self.g.dtype, device=self.g.device).index_copy(0, self.sel, self.g)
        return self.g

    def penalty(self):
        return self.pen * ((self.gains() - 1) ** 2).mean() if self.pen else 0.0

    def factored(self):
        """Vrai si le pas récurrent doit être calculé direction par direction (sélectif, rotations, gains par neurone)."""
        return bool(self.selective or self.rotations or self.neuron_gains)

    def _rotate(self, z):
        c, s = torch.cos(self.theta), torch.sin(self.theta)
        zi, zj = z[:, self.rot_i], z[:, self.rot_j]
        z = z.clone()
        z[:, self.rot_i] = c * zi - s * zj
        z[:, self.rot_j] = s * zi + c * zj
        return z

    def recurrent_step(self, x, u):
        """W(t)·x pour un lot d'états x (B, N) et d'entrées u (B, n_in), forme factorisée."""
        z = (x * self.dR if self.neuron_gains else x) @ self.Vs          # coordonnées dans les directions de W₀
        if self.rotations:
            z = self._rotate(z)
        g = self.gains()
        if self.selective:
            g = g + u @ self.G.T                                          # (B, N) : amplitudes du moment
        r = (z * (self.sig * g)) @ self.Us.T
        if self.neuron_gains:
            r = r * self.dL
        if self.rank:
            r = r + (x @ self.B) @ self.A.T
        return r

    def weights(self):
        g = self.gains()
        if self.basis == "svd":
            V = self.Vs
            if self.rotations:                                  # rotations appliquées aux directions d'entrée
                V = self._rotate(V)
            W = (self.Us * (self.sig * g)) @ V.T
            if self.neuron_gains:
                W = self.dL[:, None] * W * self.dR[None, :]
        else:
            ge = (g + g[self.pair]) / 2                         # modes conjugués : même module,
            ph = (self.phi - self.phi[self.pair]) / 2           # phases opposées -> W réelle
            d = self.lam * ge * torch.exp(1j * ph)
            W = ((self.Pm * d) @ self.Pinv).real
        if self.rank:
            W = W + self.A @ self.B.T
        return self.Win_r + self.Win_l, W

    def recurrent_params(self):
        return [p for name, p in self.named_parameters() if name != "bias" and p.requires_grad]

    def param_groups(self, lr):
        return [{"params": self.recurrent_params(), "lr": lr}]

    def forward(self, U):
        Win, W = self.weights()
        a = torch.sigmoid(self.c_leak) if self.use_leak else None
        B, L, _ = U.shape
        x = torch.zeros(B, W.shape[0], dtype=U.dtype, device=U.device)
        xs = []
        fact = self.factored()
        for t in range(L):
            rec = self.recurrent_step(x, U[:, t]) if fact else x @ W.T
            h = torch.tanh(U[:, t] @ Win.T + rec + self.bias)
            x = h if a is None else (1 - a) * x + a * h
            xs.append(x)
        return torch.stack(xs, 1)


class DeepTuned(torch.nn.Module):
    """Deux réservoirs accordés de N/2 neurones empilés ; sortie = états concaténés (N)."""

    def __init__(self, n, n_in=1, seed=0, dtype=torch.float32, device="cpu", sparsity=0.1, **kw):
        super().__init__()
        from .torch_models import make_reservoir
        h = n // 2
        Win1, W1 = make_reservoir(h, n_in=n_in, rho=RHO, in_scale=IN_SCALE, sparsity=sparsity, seed=seed)
        _, W2 = make_reservoir(h, n_in=1, rho=RHO, in_scale=IN_SCALE, sparsity=sparsity, seed=seed + 5000)
        Win2 = np.random.default_rng(seed + 6000).uniform(-1, 1, (h, h)) / np.sqrt(h)
        self.l1 = Tuned(Win1, W1, seed, dtype, device, **kw)
        self.l2 = Tuned(Win2, W2, seed + 1, dtype, device, **kw)
        self.n_features = 2 * h
        self.bias = torch.nn.Parameter(torch.zeros(1, dtype=dtype, device=device), requires_grad=False)

    def recurrent_params(self):
        return self.l1.recurrent_params() + self.l2.recurrent_params() + [self.l1.bias, self.l2.bias]

    def param_groups(self, lr):
        return [{"params": self.recurrent_params(), "lr": lr}]

    def forward(self, U):
        x1 = self.l1(U)
        return torch.cat([x1, self.l2(x1)], -1)


class FastTuned(torch.nn.Module):
    """Accordé rapide : W = P₁ · F⁻¹ · diag(λ ⊙ g) · F · P₂, coût N·log N par pas.

    F est la transformée de Fourier réelle (rfft), λ un spectre fixe tiré au hasard : modules répartis comme les
    valeurs singulières d'une matrice aléatoire, phases uniformes ; P₁ et P₂ sont des permutations aléatoires fixes,
    qui cassent la structure en anneau d'une matrice circulante. Les valeurs singulières de W sont |λ_k · g_k| :
    comme pour le réservoir accordé, on n'apprend qu'une amplitude par direction (N/2 + 1 valeurs), mais les
    directions sont celles de Fourier (structurées) au lieu d'être aléatoires.
    Rayon spectral ramené à 0.9 (calcul exact jusqu'à N = 3 000, au-delà rapport estimé à N = 2 000).
    """

    def __init__(self, Win_r, seed=0, dtype=torch.float32, device="cpu", learn=True):
        super().__init__()
        n = Win_r.shape[0]
        self.n, self.n_features = n, n
        rng = np.random.default_rng(7000 + seed)
        m = n // 2 + 1
        sv = np.linalg.svd(rng.normal(size=(min(n, 1000), min(n, 1000))), compute_uv=False)
        mod = rng.choice(sv / sv.max(), size=m)                          # profil de valeurs singulières aléatoires
        ph = rng.uniform(0, 2 * np.pi, m)
        ph[0] = 0.0
        if n % 2 == 0:
            ph[-1] = 0.0
        lam = mod * np.exp(1j * ph)
        p1, p2 = rng.permutation(n), rng.permutation(n)
        lam = lam * RHO / self._radius(lam, p1, p2, n, seed)
        cdt = torch.complex64 if dtype == torch.float32 else torch.complex128
        self.register_buffer("lam", torch.as_tensor(lam, dtype=cdt, device=device))
        self.register_buffer("p1", torch.as_tensor(p1, device=device))
        self.register_buffer("p2", torch.as_tensor(p2, device=device))
        self.register_buffer("Win_r", torch.as_tensor(np.asarray(Win_r), dtype=dtype, device=device))
        P = torch.nn.Parameter
        self.g = P(torch.ones(m, dtype=dtype, device=device), requires_grad=learn)
        self.Win_l = P(torch.zeros_like(self.Win_r), requires_grad=learn)
        self.bias = P(torch.zeros(n, dtype=dtype, device=device), requires_grad=learn)

    @staticmethod
    def dense(lam, p1, p2, n):
        """Matrice W explicite (pour les tests et le calcul du rayon spectral)."""
        I = np.eye(n)
        Z = np.fft.rfft(I[p2], axis=0)                                  # colonnes : F · P₂ · e_j
        C = np.fft.irfft(lam[:, None] * Z, n=n, axis=0)
        W = np.empty_like(C)
        W[p1] = C                                                        # out[p1] = C  <=>  P₁ appliqué
        return W

    @classmethod
    def _radius(cls, lam, p1, p2, n, seed):
        if n <= 3000:
            return _spectral_radius(cls.dense(lam, p1, p2, n))
        # au-delà : rapport rayon spectral / plus grand module mesuré sur un tirage de taille 2 000
        rng = np.random.default_rng(7000 + seed)
        m = 1001
        sv = np.linalg.svd(rng.normal(size=(1000, 1000)), compute_uv=False)
        mod = rng.choice(sv / sv.max(), size=m)
        ph = rng.uniform(0, 2 * np.pi, m); ph[0] = ph[-1] = 0.0
        q1, q2 = rng.permutation(2000), rng.permutation(2000)
        small = mod * np.exp(1j * ph)
        return np.abs(lam).max() * _spectral_radius(cls.dense(small, q1, q2, 2000)) / np.abs(small).max()

    def matvec(self, x):
        """W·x pour un lot (B, N) : permutation, FFT, gains, FFT inverse, permutation."""
        z = torch.fft.rfft(x[:, self.p2], dim=-1) * (self.lam * self.g)
        y = torch.fft.irfft(z, n=self.n, dim=-1)
        out = torch.empty_like(y)
        out[:, self.p1] = y
        return out

    def recurrent_params(self):
        return [p for name, p in self.named_parameters() if name != "bias" and p.requires_grad]

    def param_groups(self, lr):
        ps = self.recurrent_params()
        return [{"params": ps, "lr": lr}] if ps else []

    def forward(self, U):
        Win = self.Win_r + self.Win_l
        B, L, _ = U.shape
        x = torch.zeros(B, self.n, dtype=U.dtype, device=U.device)
        xs = []
        for t in range(L):
            x = torch.tanh(U[:, t] @ Win.T + self.matvec(x) + self.bias)
            xs.append(x)
        return torch.stack(xs, 1)


class GatedTuned(torch.nn.Module):
    """Partie 5 — réservoir accordé à portes : des cellules GRU ou LSTM dont chaque matrice récurrente est accordée.

    Toutes les matrices récurrentes partagent les directions aléatoires figées du réservoir, W₀ = U·diag(σ)·Vᵀ ;
    chacune n'apprend que son profil d'amplitudes : W_k = U·diag(σ ⊙ g_k)·Vᵀ (N paramètres par matrice au lieu
    de N²). Les portes ont aussi des poids d'entrée et des biais appris.

        cell="gru"       z = σ(W_z h + B_z u + c_z),  r = σ(W_r h + B_r u + c_r)
                         h̃ = tanh(W (r ⊙ h) + W_in u + b),  h ← (1 − z) ⊙ h + z ⊙ h̃
        cell="gru_light" même chose, mais portes neurone par neurone : W_z h remplacé par a_z ⊙ h
        cell="lstm"      f, i, o = σ(W_k h + B_k u + c_k),  c̃ = tanh(W h + W_in u + b)
                         c ← f ⊙ c + i ⊙ c̃,  h = o ⊙ tanh(c)
        cell="mingru"    porte de mise à jour qui ne dépend que de l'entrée (comme minGRU) : z = σ(B_z u + c_z),
                         h ← (1 − z) ⊙ h + z ⊙ tanh(W h + W_in u + b)

    Départ : amplitudes des portes nulles et biais des portes à ±4 (portes presque ouvertes ou fermées), si bien que
    la cellule part presque exactement du réservoir accordé : z ≈ r ≈ 0.98 pour le GRU ; f ≈ 0.02, i ≈ o ≈ 0.98
    pour le LSTM.
    """

    GATES = {"gru": ("z", "r"), "gru_light": ("z", "r"), "lstm": ("f", "i", "o"), "mingru": ("z",)}
    BIAS0 = {"z": 4.0, "r": 4.0, "f": -4.0, "i": 4.0, "o": 4.0}

    def __init__(self, Win_r, W_r, seed=0, dtype=torch.float32, device="cpu", cell="gru", gates_only=False):
        super().__init__()
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)   # noqa: E731
        P = torch.nn.Parameter
        n, n_in = W_r.shape[0], Win_r.shape[1]
        self.cell, self.n_features = cell, n
        U_, sig, Vt = np.linalg.svd(np.asarray(W_r, dtype=float))
        self.register_buffer("Us", f(U_)); self.register_buffer("sig", f(sig)); self.register_buffer("Vs", f(Vt.T))
        self.register_buffer("Win_r", f(Win_r))
        # amplitudes de la matrice principale ; gates_only : figées (contrôle « seules les portes apprennent »,
        # comme les Gated Echo State Networks de Di Sarli et al., 2022)
        self.g = P(torch.ones(n, dtype=dtype, device=device), requires_grad=not gates_only)
        self.Win_l = P(torch.zeros_like(f(Win_r)))
        self.bias = P(torch.zeros(n, dtype=dtype, device=device))
        for k in self.GATES[cell]:
            if cell != "mingru":
                if cell == "gru_light":
                    setattr(self, f"a_{k}", P(torch.zeros(n, dtype=dtype, device=device)))
                else:
                    setattr(self, f"g_{k}", P(torch.zeros(n, dtype=dtype, device=device)))
            setattr(self, f"B_{k}", P(torch.zeros(n, n_in, dtype=dtype, device=device)))
            setattr(self, f"c_{k}", P(torch.full((n,), self.BIAS0[k], dtype=dtype, device=device)))

    def recurrent_params(self):
        return [p for name, p in self.named_parameters() if name != "bias" and p.requires_grad]

    def param_groups(self, lr):
        return [{"params": self.recurrent_params(), "lr": lr}]

    def _rec(self, zdir, gains):
        """W_k · h à partir des coordonnées zdir = h·V dans les directions du réservoir."""
        return (zdir * (self.sig * gains)) @ self.Us.T

    def _gate(self, k, h, zdir, u):
        if self.cell == "mingru":
            pre = 0.0
        elif self.cell == "gru_light":
            pre = getattr(self, f"a_{k}") * h
        else:
            pre = self._rec(zdir, getattr(self, f"g_{k}"))
        return torch.sigmoid(pre + u @ getattr(self, f"B_{k}").T + getattr(self, f"c_{k}"))

    def forward(self, U):
        Win = self.Win_r + self.Win_l
        B, L, _ = U.shape
        n = self.n_features
        h = torch.zeros(B, n, dtype=U.dtype, device=U.device)
        c = torch.zeros_like(h)
        hs = []
        for t in range(L):
            u = U[:, t]
            zdir = h @ self.Vs
            inp = u @ Win.T + self.bias
            if self.cell in ("gru", "gru_light"):
                z, r = self._gate("z", h, zdir, u), self._gate("r", h, zdir, u)
                cand = torch.tanh(self._rec((r * h) @ self.Vs, self.g) + inp)
                h = (1 - z) * h + z * cand
            elif self.cell == "mingru":
                z = self._gate("z", h, zdir, u)
                h = (1 - z) * h + z * torch.tanh(self._rec(zdir, self.g) + inp)
            else:                                                    # lstm
                fg, ig, og = (self._gate(k, h, zdir, u) for k in ("f", "i", "o"))
                c = fg * c + ig * torch.tanh(self._rec(zdir, self.g) + inp)
                h = og * torch.tanh(c)
            hs.append(h)
        return torch.stack(hs, 1)
