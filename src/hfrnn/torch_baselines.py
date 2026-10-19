"""Modèles récurrents existants, pour la comparaison de la partie 5.

Ils sont entraînés exactement comme nos modèles (torch_models.train_regression) : mêmes fenêtres, Adam,
gradient clippé à 1, readout de départ nul, readout final en ridge sur les états. Taille d'état égale :
N unités cachées pour le GRU et le LSTM, N caractéristiques en sortie pour le LRU.

    "gru"   torch.nn.GRU, une couche, N unités (≈ 3 N² paramètres récurrents)
    "lstm"  torch.nn.LSTM, une couche, N unités (≈ 4 N² paramètres récurrents)
            rec_init="ortho" : initialisation soignée habituelle (poids récurrents orthogonaux, biais de la
            porte d'oubli du LSTM à 1) au lieu de l'initialisation par défaut de PyTorch.
    "lru"   Linear Recurrent Unit (Orvieto et al., 2023), une couche :
                h(t) = λ ⊙ h(t−1) + γ ⊙ B u(t)     (h complexe de dimension N/2, λ diagonale apprise)
                y(t) = GELU( Re(C h(t)) + D u(t) )   (N caractéristiques, readout ridge dessus)
            λ = exp(−exp(ν) + i·exp(θ)), |λ| tiré dans [r_min, r_max], γ = sqrt(1 − |λ|²).
            Par défaut une couche (≈ N² paramètres) ; layers=2 empile deux couches avec une connexion
            résiduelle, x₂ = GELU(LRU₂(x₁)) + x₁, comme dans l'article (sans normalisation).
            tuned=True : « réservoir accordé linéaire » (partie 5). C et D restent aléatoires et figés ; seules
            les valeurs propres λ, la normalisation γ et les entrées B apprennent (≈ 2,5 N paramètres au lieu de N²).

Le LRU est écrit en arithmétique réelle (partie réelle et imaginaire séparées), pour éviter les
complexes dans l'autograd.
"""
import math

import numpy as np
import torch


class Baseline(torch.nn.Module):
    CHUNK = 8192                                           # longueur max d'un appel cuDNN (GRU, LSTM)

    def __init__(self, kind, n, n_in=1, seed=0, dtype=torch.float32, device="cpu",
                 r_min=0.5, r_max=0.99, max_phase=math.pi, rec_init="default", layers=1, tuned=False):
        super().__init__()
        self.kind, self.n_features = kind, n
        torch.manual_seed(seed)
        if kind in ("gru", "lstm"):
            cls = torch.nn.GRU if kind == "gru" else torch.nn.LSTM
            self.rnn = cls(n_in, n, batch_first=True).to(device, dtype)
            if rec_init == "ortho":
                with torch.no_grad():
                    W = self.rnn.weight_hh_l0                  # portes empilées : (k·N, N)
                    for k in range(W.shape[0] // n):
                        torch.nn.init.orthogonal_(W[k * n:(k + 1) * n])
                    if kind == "lstm":                         # ordre des portes : entrée, oubli, cellule, sortie
                        self.rnn.bias_ih_l0[n:2 * n] = 1.0
                        self.rnn.bias_hh_l0[n:2 * n] = 0.0
        elif kind == "lru":
            self.layers = torch.nn.ModuleList(
                [LRULayer(n_in if i == 0 else n, n, 3000 + 17 * i + seed, dtype, device, r_min, r_max, max_phase)
                 for i in range(layers)])
            if tuned:
                # réservoir accordé linéaire : les directions de sortie C et le passage direct D restent tirés
                # au hasard et figés ; seules les valeurs propres λ (module et phase), la normalisation γ et les
                # entrées B apprennent, comme le réservoir accordé apprend ses amplitudes et ses entrées.
                # Au-delà de la première couche, l'entrée B (N × N/2) est elle aussi figée : seules λ et γ y apprennent.
                for i, layer in enumerate(self.layers):
                    for name in ("C_re", "C_im", "D") + (("B_re", "B_im") if i > 0 else ()):
                        getattr(layer, name).requires_grad_(False)
        else:
            raise ValueError(kind)
        # pas de biais séparé : train_regression l'ignore s'il n'est pas appris
        self.bias = torch.nn.Parameter(torch.zeros(1, dtype=dtype, device=device), requires_grad=False)

    def recurrent_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_groups(self, lr):
        return [{"params": self.recurrent_params(), "lr": lr}]

    def forward(self, U):
        """U : (B, L, n_in) -> caractéristiques (B, L, N), état initial nul."""
        if self.kind in ("gru", "lstm"):
            # cuDNN refuse les très longues séquences d'un seul tenant (CUDNN_STATUS_NOT_SUPPORTED, vu sur
            # Silverbox, 65 000 pas) : on découpe en morceaux et on passe l'état caché d'un morceau au suivant,
            # ce qui donne exactement le même résultat.
            U = U.contiguous()
            if U.shape[1] <= self.CHUNK:
                return self.rnn(U)[0]
            outs, h = [], None
            for s in range(0, U.shape[1], self.CHUNK):
                o, h = self.rnn(U[:, s:s + self.CHUNK].contiguous(), h)
                outs.append(o)
            return torch.cat(outs, 1)
        x = U
        for i, layer in enumerate(self.layers):
            y = layer(x)
            x = y if i == 0 else y + x                      # connexion résiduelle entre couches
        return x


class LRULayer(torch.nn.Module):
    """Une couche LRU : h(t) = λ ⊙ h(t−1) + γ ⊙ B u(t), sortie GELU(Re(C h) + D u)."""

    def __init__(self, n_in, n, seed, dtype, device, r_min, r_max, max_phase):
        super().__init__()
        ns = max(1, n // 2)
        rng = np.random.default_rng(seed)
        u1, u2 = rng.uniform(size=ns), rng.uniform(size=ns)
        nu = np.log(-0.5 * np.log(u1 * (r_max ** 2 - r_min ** 2) + r_min ** 2))
        theta = np.log(max_phase * u2 + 1e-6)
        mod = np.exp(-np.exp(nu))
        f = lambda a: torch.nn.Parameter(torch.as_tensor(a, dtype=dtype, device=device))   # noqa: E731
        self.nu_log, self.theta_log = f(nu), f(theta)
        self.gamma_log = f(np.log(np.sqrt(1 - mod ** 2)))
        self.B_re = f(rng.normal(0, 1 / np.sqrt(2 * n_in), (ns, n_in)))
        self.B_im = f(rng.normal(0, 1 / np.sqrt(2 * n_in), (ns, n_in)))
        self.C_re = f(rng.normal(0, 1 / np.sqrt(ns), (n, ns)))
        self.C_im = f(rng.normal(0, 1 / np.sqrt(ns), (n, ns)))
        self.D = f(rng.normal(0, 1, (n, n_in)))

    def forward(self, U):
        r = torch.exp(-torch.exp(self.nu_log))
        th = torch.exp(self.theta_log)
        lr_, li = r * torch.cos(th), r * torch.sin(th)
        g = torch.exp(self.gamma_log)
        bu_re = (U @ self.B_re.T) * g                      # (B, L, ns)
        bu_im = (U @ self.B_im.T) * g
        B, L, _ = U.shape
        hr = torch.zeros(B, lr_.numel(), dtype=U.dtype, device=U.device)
        hi = torch.zeros_like(hr)
        hs_r, hs_i = [], []
        for t in range(L):
            hr, hi = lr_ * hr - li * hi + bu_re[:, t], lr_ * hi + li * hr + bu_im[:, t]
            hs_r.append(hr); hs_i.append(hi)
        Hr, Hi = torch.stack(hs_r, 1), torch.stack(hs_i, 1)
        y = Hr @ self.C_re.T - Hi @ self.C_im.T + U @ self.D.T
        return torch.nn.functional.gelu(y)
