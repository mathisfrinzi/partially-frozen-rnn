"""Partie 5 — réservoir cisaillé à portes : une cellule de type GRU dont les matrices récurrentes sont, selon le choix,
figées (tirage aléatoire), cisaillées en bande (partie triangulaire apprise sur k sur-diagonales), apprises en entier ou
réduites à une diagonale.

Une cellule GRU a trois blocs de poids : la porte de mise à jour z, la porte de réinitialisation r et le candidat c. Chacun
porte une matrice récurrente N×N, que `maps = (z, r, c)` règle :

    "shear"   Q·(R + S) : décomposition QR d'un tirage aléatoire figé, S strictement triangulaire limitée aux `band`
              premières sur-diagonales (≈ band·N paramètres) ; avec learn_g, une amplitude par ligne en plus
    "full"    tirage aléatoire figé + correction apprise entière (N² paramètres)
    "frozen"  tirage aléatoire figé, aucun paramètre
    "diag"    portes neurone par neurone, a ⊙ h (N paramètres) ; pour les portes seulement

Pour le candidat, le tirage est W₀ (le réservoir de départ). Pour les portes, c'est un autre réservoir de même densité et
de même rayon spectral, multiplié par `gate_scale`.

    cell="gru"       z = σ(M_z h + B_z u + c_z),  r = σ(M_r h + B_r u + c_r)
                     h̃ = tanh(M_c (r ⊙ h) + W_in u + b),   h ← (1 − z) ⊙ h + z ⊙ h̃
    cell="mingru"    la porte de mise à jour ne dépend que de l'entrée : z = σ(B_z u + c_z)
    cell="lin_gate"  récurrence linéaire à porte d'entrée : h ← (1 − z) ⊙ (M_c h) + z ⊙ (W_in u + b),
                     lecture GELU(h) comme dans le LRU
    cell="linear"    h ← M_c h + W_in u + b, lecture GELU(h)

`train` choisit ce qui apprend : "in" = entrées W_in, biais b et entrées des portes B ; "gate" = matrices récurrentes et
biais des portes ; "rec" = la matrice du candidat. Le reste reste figé.

Départ : les portes sont presque ouvertes (biais +4) pour gru et mingru, si bien que la cellule démarre très près d'un
réservoir. Pour lin_gate, le biais de la porte vaut −2.
"""
import numpy as np
import torch


def _random_reservoir(n, density, rho, rng):
    W = rng.uniform(-1, 1, (n, n)) * (rng.random((n, n)) < density)
    r = max(abs(np.linalg.eigvals(W)))
    return W * (rho / r if r > 0 else 1.0)


class ShearMap(torch.nn.Module):
    """Une matrice récurrente N×N dont une partie est figée et une partie est apprise (voir `maps`)."""

    def __init__(self, W0, kind, band, learn_g, scale, dtype, device):
        super().__init__()
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)   # noqa: E731
        n = W0.shape[0]
        self.kind, self.scale = kind, float(scale)
        if kind == "shear":
            Q, R = np.linalg.qr(np.asarray(W0, float))
            s = np.where(np.diag(R) >= 0, 1.0, -1.0)
            Q, R = Q * s, R * s[:, None]
            tri = torch.triu_indices(n, n, 1)
            if band:
                tri = tri[:, (tri[1] - tri[0]) <= int(band)]
            self.register_buffer("tri", tri.to(device))
            self.register_buffer("Q", f(Q)); self.register_buffer("R", f(R))
            self.S = torch.nn.Parameter(torch.zeros(tri.shape[1], dtype=dtype, device=device))
            self.g = torch.nn.Parameter(torch.ones(n, dtype=dtype, device=device)) if learn_g else None
        elif kind in ("full", "frozen"):
            self.register_buffer("W0", f(W0))
            if kind == "full":
                self.D = torch.nn.Parameter(torch.zeros(n, n, dtype=dtype, device=device))
        elif kind == "diag":
            self.a = torch.nn.Parameter(torch.zeros(n, dtype=dtype, device=device))
        else:
            raise ValueError(kind)
        self.n = n

    def matrix(self):
        """La matrice N×N, ou None pour une porte diagonale."""
        if self.kind == "shear":
            T = torch.zeros(self.n, self.n, dtype=self.S.dtype, device=self.S.device)
            T[self.tri[0], self.tri[1]] = self.S
            M = self.R + T
            if self.g is not None:
                M = M * self.g[:, None]
            return self.scale * (self.Q @ M)
        if self.kind == "full":
            return self.scale * (self.W0 + self.D)
        if self.kind == "frozen":
            return self.scale * self.W0
        return None

    def apply(self, M, h):
        return self.a * h if self.kind == "diag" else h @ M.T


class ShearGated(torch.nn.Module):
    GATES = {"gru": ("z", "r"), "mingru": ("z",), "lin_gate": ("z",), "linear": ()}

    def __init__(self, Win_r, W_r, seed=0, dtype=torch.float32, device="cpu", cell="gru",
                 maps=("shear", "shear", "shear"), band=10, learn_g=False, train=("in", "gate", "rec"),
                 gate_scale=0.25):
        super().__init__()
        f = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype, device=device)   # noqa: E731
        P = torch.nn.Parameter
        W_r = np.asarray(W_r, float)
        n, n_in = W_r.shape[0], Win_r.shape[1]
        self.cell, self.n_features = cell, n
        mz, mr, mc = maps
        density = float((W_r != 0).mean())
        rho = float(max(abs(np.linalg.eigvals(W_r))))
        self.names = self.GATES[cell]
        self.maps = torch.nn.ModuleDict()
        for i, k in enumerate(self.names if cell == "gru" else ()):     # seule la cellule gru a des portes récurrentes
            kind = (mz, mr)[i]
            W0 = _random_reservoir(n, density, rho, np.random.default_rng(7000 + 10 * seed + i))
            self.maps[k] = ShearMap(W0, kind, band, learn_g, gate_scale if kind != "diag" else 1.0, dtype, device)
        self.cand = ShearMap(W_r, mc, band, learn_g, 1.0, dtype, device)
        self.register_buffer("Win_r", f(Win_r))
        self.Win_l = P(torch.zeros_like(f(Win_r)))
        self.bias = P(torch.zeros(n, dtype=dtype, device=device))
        c0 = -2.0 if cell == "lin_gate" else 4.0
        for k in self.names:
            setattr(self, f"B_{k}", P(torch.zeros(n, n_in, dtype=dtype, device=device)))
            setattr(self, f"c_{k}", P(torch.full((n,), c0, dtype=dtype, device=device)))
        self._groups = {"in": [self.Win_l, self.bias] + [getattr(self, f"B_{k}") for k in self.names],
                        "gate": [p for k in self.maps for p in self.maps[k].parameters()]
                                + [getattr(self, f"c_{k}") for k in self.names],
                        "rec": list(self.cand.parameters())}
        for grp, ps in self._groups.items():
            for p in ps:
                p.requires_grad_(grp in train)

    def recurrent_params(self):
        return [p for name, p in self.named_parameters() if name != "bias" and p.requires_grad]

    def param_groups(self, lr):
        ps = self.recurrent_params()
        return [{"params": ps, "lr": lr}] if ps else []

    def forward(self, U):
        Win = self.Win_r + self.Win_l
        B, L, _ = U.shape
        n = self.n_features
        Mc = self.cand.matrix()
        Mg = {k: self.maps[k].matrix() for k in self.maps}
        h = torch.zeros(B, n, dtype=U.dtype, device=U.device)
        inputs = U @ Win.T + self.bias                               # (B, L, N), calculé une fois
        gin = {k: U @ getattr(self, f"B_{k}").T + getattr(self, f"c_{k}") for k in self.names}
        hs = []
        for t in range(L):
            inp = inputs[:, t]
            if self.cell == "gru":
                z = torch.sigmoid(self.maps["z"].apply(Mg["z"], h) + gin["z"][:, t])
                r = torch.sigmoid(self.maps["r"].apply(Mg["r"], h) + gin["r"][:, t])
                cand = torch.tanh(self.cand.apply(Mc, r * h) + inp)
                h = (1 - z) * h + z * cand
            elif self.cell == "mingru":
                z = torch.sigmoid(gin["z"][:, t])
                h = (1 - z) * h + z * torch.tanh(self.cand.apply(Mc, h) + inp)
            elif self.cell == "lin_gate":
                z = torch.sigmoid(gin["z"][:, t])
                h = (1 - z) * self.cand.apply(Mc, h) + z * inp
            else:                                                    # linear
                h = self.cand.apply(Mc, h) + inp
            hs.append(h)
        X = torch.stack(hs, 1)
        return torch.nn.functional.gelu(X) if self.cell in ("lin_gate", "linear") else X
