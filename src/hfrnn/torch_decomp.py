"""Partie 2 — d'autres décompositions de la matrice du réservoir : Iwasawa (KAN) et Jordan-Chevalley.

Le réservoir accordé apprend les amplitudes de la décomposition en valeurs singulières W₀ = U·diag(σ)·Vᵀ. On essaie
deux autres décompositions, chacune avec une pièce que l'accordé n'apprend pas.

    Iwasawa (K·A·N, c'est-à-dire QR)   W₀ = Q·diag(a)·N₀ : Q orthogonale (figée), a > 0 les amplitudes, N₀ triangulaire
                                       supérieure à diagonale unité (le « cisaillement »).
        iwasawa_A    W = Q·diag(g)·R           apprend g (N paramètres)
        iwasawa_N    W = Q·(R + S)                  apprend S, partie strictement triangulaire (N(N−1)/2 paramètres)
        iwasawa_AN   W = Q·diag(g)·(R + S)           apprend les deux (R = diag(a)·N₀)

    Jordan-Chevalley (semi-simple + nilpotent). Une matrice aléatoire est presque sûrement diagonalisable, donc sa
    partie nilpotente est nulle : la partie semi-simple apprise est la variante « valeurs propres » de l'accordé.
        jordan_S     W = Re(P·diag(λ ⊙ g·e^{iφ})·P⁻¹)                 module et phase de chaque mode (2 N paramètres)
        jordan_SN    W = (partie semi-simple) + Qr·T·Qrᵀ               T strictement triangulaire apprise, Qr orthogonale
                     aléatoire figée : la matrice Qr·T·Qrᵀ est nilpotente. La commutation avec la partie
                     semi-simple, exigée par le théorème, n'est pas imposée : c'est une variante « inspirée de ».

    band=k (optionnel) : la partie triangulaire apprise (S ou T) est limitée aux k premières sur-diagonales, soit environ
    k·N paramètres au lieu de N(N−1)/2 : le « réservoir cisaillé en bande » et le « réservoir nilpotent en bande ».
    Contrôles « sans la base Q » (avec band = k, même nombre de coefficients appris que la bande) :
        direct       W = W₀ + M, M a ses coefficients appris à des positions tirées au hasard dans toute la matrice, dans
                     la base des neurones (aucune décomposition)
        direct_tri   idem, positions tirées dans le triangle strictement supérieur de W₀ lui-même (pas celui de R)
    scatter=True (avec band) : contrôle « cisaillé creux » : le même nombre de coefficients, mais à des positions tirées au
    hasard dans tout le triangle supérieur, pour savoir si c'est la bande qui compte ou seulement le nombre de coefficients.
"""
import numpy as np
import torch

from .torch_tuned import Tuned


class Decomp(torch.nn.Module):
    def __init__(self, Win_r, W_r, form="iwasawa_A", seed=0, dtype=torch.float32, device="cpu", band=None, scatter=False):
        super().__init__()
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)   # noqa: E731
        P = torch.nn.Parameter
        n = W_r.shape[0]
        self.form, self.n_features = form, n
        tri = torch.triu_indices(n, n, 1)                                                     # indices strictement triangulaires
        if band:                                                                              # k premières sur-diagonales
            keep = (tri[1] - tri[0]) <= int(band)
            if scatter:                                                                       # même nombre, positions au hasard
                pick = np.random.default_rng(9000 + seed).permutation(tri.shape[1])[:int(keep.sum())]
                tri = tri[:, torch.as_tensor(np.sort(pick))]
            else:
                tri = tri[:, keep]
        if form in ("direct", "direct_tri"):                                                  # mêmes nombres, base des neurones
            count = int(((torch.triu_indices(n, n, 1)[1] - torch.triu_indices(n, n, 1)[0]) <= int(band)).sum()) if band \
                else n * (n - 1) // 2
            cand = torch.triu_indices(n, n, 1) if form == "direct_tri" else \
                torch.stack(torch.meshgrid(torch.arange(n), torch.arange(n), indexing="ij")).reshape(2, -1)
            pick = np.random.default_rng(9100 + seed).permutation(cand.shape[1])[:count]
            tri = cand[:, torch.as_tensor(np.sort(pick))]
        self.register_buffer("tri", tri.to(device))
        if form in ("direct", "direct_tri"):
            self.register_buffer("Win_r", f(Win_r))
            self.Win_l = P(torch.zeros_like(f(Win_r)))
            self.bias = P(torch.zeros(n, dtype=dtype, device=device))
            self.register_buffer("W0", f(W_r))
            self.S = P(torch.zeros(self.tri.shape[1], dtype=dtype, device=device))
        elif form.startswith("iwasawa"):
            self.register_buffer("Win_r", f(Win_r))
            self.Win_l = P(torch.zeros_like(f(Win_r)))
            self.bias = P(torch.zeros(n, dtype=dtype, device=device))
            Q, R = np.linalg.qr(np.asarray(W_r, float))
            s = np.where(np.diag(R) >= 0, 1.0, -1.0)
            Q, R = Q * s, R * s[:, None]                         # diagonale de R positive
            self.register_buffer("Q", f(Q)); self.register_buffer("R", f(R))   # R = diag(a)·N₀ (sans division : a peut être nul)
            learn = form.split("_")[1]
            if "A" in learn:
                self.g = P(torch.ones(n, dtype=dtype, device=device))
            if "N" in learn:
                self.S = P(torch.zeros(self.tri.shape[1], dtype=dtype, device=device))
        elif form in ("jordan_S", "jordan_SN"):
            self.inner = Tuned(Win_r, W_r, seed=seed, dtype=dtype, device=device, basis="eig")
            self.bias = self.inner.bias                           # même paramètre
            _, S0 = self.inner.weights()
            self.register_buffer("Nil0", (f(W_r) - S0.detach()))   # partie nilpotente de W₀ (≈ 0 si W₀ est dense)
            if form == "jordan_SN":
                Qr, _ = np.linalg.qr(np.random.default_rng(5000 + seed).normal(size=(n, n)))
                self.register_buffer("Qr", f(Qr))
                self.T = P(torch.zeros(self.tri.shape[1], dtype=dtype, device=device))
        else:
            raise ValueError(form)

    def recurrent_params(self):
        return [p for k, p in self.named_parameters() if k != "bias" and k != "inner.bias"]

    def param_groups(self, lr):
        return [{"params": self.recurrent_params(), "lr": lr}]

    def _tri(self, v):
        """Matrice N×N dont les entrées aux positions self.tri sont le vecteur v (strictement triangulaire sauf « direct »)."""
        n = self.n_features
        M = torch.zeros(n, n, dtype=v.dtype, device=v.device)
        M[self.tri[0], self.tri[1]] = v
        return M

    def weights(self):
        if self.form in ("direct", "direct_tri"):
            return self.Win_r + self.Win_l, self.W0 + self._tri(self.S)
        if self.form.startswith("iwasawa"):
            g = self.g if hasattr(self, "g") else 1.0
            R = self.R + (self._tri(self.S) if hasattr(self, "S") else 0.0)
            return self.Win_r + self.Win_l, self.Q @ (R * (g if isinstance(g, float) else g[:, None]))
        Win, W = self.inner.weights()
        W = W + self.Nil0
        if self.form == "jordan_SN":
            W = W + self.Qr @ self._tri(self.T) @ self.Qr.T
        return Win, W

    def forward(self, U):
        Win, W = self.weights()
        bias = self.bias
        B, L, _ = U.shape
        x = torch.zeros(B, W.shape[0], dtype=U.dtype, device=U.device)
        xs = []
        for t in range(L):
            x = torch.tanh(U[:, t] @ Win.T + x @ W.T + bias)
            xs.append(x)
        return torch.stack(xs, 1)
