"""Partie 9, étapes 2 et 3 — réservoir dont la matrice apprend dans un sous-espace choisi à partir du gradient.

    W = W₀ + reshape(θ·V)        V : (r, N²) base orthonormée du sous-espace (directions de matrices), θ : r coefficients
    W = W₀ + M                   M : r coefficients aux positions plates `idx` (sous-espace de coordonnées)

thaw (optionnel) : pas de dégel de chaque coordonnée (dégel progressif, experiments/47_protocole_lie.py).

Comme les autres structures, W_in et les biais restent appris (et la lecture est ajustée à part, hors du modèle). Seul
change le choix des r directions : au lieu d'une décomposition fixée d'avance (accordé, tourné, cisaillé), elles sont
tirées du gradient mesuré sur les données de la tâche (voir experiments/46_sous_espace_appris.py).
"""
import numpy as np
import torch


class Subspace(torch.nn.Module):
    def __init__(self, Win_r, W_r, V=None, idx=None, seed=0, dtype=torch.float32, device="cpu", thaw=None):
        super().__init__()
        if (V is None) == (idx is None):
            raise ValueError("donner V (directions) ou idx (positions), pas les deux")
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)   # noqa: E731
        n = W_r.shape[0]
        self.n_features = n
        self.register_buffer("Win_r", f(Win_r))
        self.register_buffer("W0", f(W_r))
        self.Win_l = torch.nn.Parameter(torch.zeros_like(f(Win_r)))
        self.bias = torch.nn.Parameter(torch.zeros(n, dtype=dtype, device=device))
        if V is not None:
            self.register_buffer("V", f(V))
            self.theta = torch.nn.Parameter(torch.zeros(self.V.shape[0], dtype=dtype, device=device))
        else:
            self.register_buffer("idx", torch.as_tensor(np.asarray(idx), dtype=torch.long, device=device))
            self.theta = torch.nn.Parameter(torch.zeros(len(self.idx), dtype=dtype, device=device))
        self._step = 0
        if thaw is not None:                               # dégel : la coordonnée j ne bouge qu'à partir du pas thaw[j]
            self.register_buffer("thaw", torch.as_tensor(np.asarray(thaw), dtype=torch.long, device=device))
            self.theta.register_hook(self._mask)

    def recurrent_params(self):
        return [self.Win_l, self.theta]

    def param_groups(self, lr):
        return [{"params": self.recurrent_params(), "lr": lr}]

    def weights(self):
        n = self.n_features
        if hasattr(self, "V"):
            M = (self.theta @ self.V).reshape(n, n)
        else:
            M = torch.zeros(n * n, dtype=self.theta.dtype, device=self.theta.device).index_add(0, self.idx, self.theta)
            M = M.reshape(n, n)
        return self.Win_r + self.Win_l, self.W0 + M

    def _mask(self, g):
        return g * (self.thaw <= self._step - 1).to(g.dtype)         # _step a déjà compté le pas en cours

    def forward(self, U):
        if self.training and torch.is_grad_enabled():                # un pas d'entraînement = un appel avec gradient
            self._step += 1
        Win, W = self.weights()
        B, L, _ = U.shape
        x = torch.zeros(B, W.shape[0], dtype=U.dtype, device=U.device)
        xs = []
        for t in range(L):
            x = torch.tanh(U[:, t] @ Win.T + x @ W.T + self.bias)
            xs.append(x)
        return torch.stack(xs, 1)
