"""
Galerie de la partie 5.4 : ce que l'entraînement change dans une récurrence linéaire de type LRU, une ligne par modèle.

Même tâche (NARMA10), 40 neurones (20 modes complexes), vrais entraînements (version PyTorch, 1 500 pas). Lignes :
  - LRU complet : tout apprend (valeurs propres λ, normalisation γ, entrées B, sorties C, passage direct D) ;
  - réservoir accordé linéaire : C et D restent tirés au hasard et figés ; seuls λ, γ et B apprennent ;
  - réservoir accordé (non linéaire, partie 3), pour comparaison : ses valeurs propres sont celles de U·diag(σ ⊙ g)·Vᵀ.
Pour chaque ligne, quatre panneaux :
  - les valeurs propres dans le plan complexe, avant (gris) et après (couleur) l'entraînement, reliées par une flèche ;
  - la mémoire de chaque mode : sa demi-vie (en pas), avant et après ;
  - ce qui a bougé : la variation relative de chaque groupe de paramètres, ‖θ − θ₀‖ / ‖θ₀‖ ;
  - la prédiction sur 120 pas de la série de test.

    python experiments/32_linear_gallery.py      (~5 min sur CPU)  -> results/figures/linear_gallery.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from hfrnn import core, tasks
from hfrnn import torch_models as TM

N, TASK, SEED = 40, "NARMA10", 0
T0, LEN = 3200, 120
ROWS = [
    ("LRU complet", "h(t) = λ ⊙ h(t−1) + γ ⊙ B·u(t),  sortie GELU(Re(C·h) + D·u) : λ, γ, B, C, D appris (≈ N² paramètres)",
     {"kind": "lru"}, "tab:pink"),
    ("Réservoir accordé linéaire (type LRU)",
     "même cellule ; C et D tirés au hasard et figés : seuls λ (module et phase), γ et B apprennent (≈ 2,5 N paramètres)",
     {"kind": "linear_tuned"}, "tab:green"),
    ("Réservoir accordé (non linéaire, partie 3), pour comparaison",
     "x(t+1) = tanh(W_in·u + U·diag(σ ⊙ g)·Vᵀ·x + b) : valeurs propres de la matrice accordée",
     {"kind": "spectrum"}, "tab:blue"),
]


def lam_lru(model):
    """Valeurs propres complexes de la première couche d'un LRU."""
    L = model.layers[0]
    with torch.no_grad():
        r = torch.exp(-torch.exp(L.nu_log)); th = torch.exp(L.theta_log)
    return (r * torch.exp(1j * th.to(torch.complex64 if r.dtype == torch.float32 else torch.complex128))).cpu().numpy()


def lam_tuned(model):
    with torch.no_grad():
        W = (model.Us * (model.sig * model.g)) @ model.Vs.T
    return np.linalg.eigvals(W.cpu().numpy())


def eig(model, kind):
    return lam_lru(model) if kind in ("lru", "linear_tuned") else lam_tuned(model)


def groups(model, kind):
    """Groupes de paramètres (valeurs actuelles), pour mesurer ce qui a bougé."""
    named = {k: v.detach().cpu().numpy().copy() for k, v in model.named_parameters()}
    if kind in ("lru", "linear_tuned"):
        p = "layers.0."
        return {"λ (module, phase)": np.concatenate([named[p + "nu_log"], named[p + "theta_log"]]),
                "γ": named[p + "gamma_log"],
                "B (entrées)": np.concatenate([named[p + "B_re"].ravel(), named[p + "B_im"].ravel()]),
                "C (sorties)": np.concatenate([named[p + "C_re"].ravel(), named[p + "C_im"].ravel()]),
                "D (direct)": named[p + "D"].ravel()}
    win = (model.Win_r + model.Win_l).detach().cpu().numpy().ravel()   # W_in = partie aléatoire + partie apprise
    return {"g (amplitudes)": named["g"], "W_in (entrées)": win}


def trainable(model, kind):
    names = {k for k, p in model.named_parameters() if p.requires_grad}
    if kind in ("lru", "linear_tuned"):
        m = {"λ (module, phase)": "layers.0.nu_log", "γ": "layers.0.gamma_log", "B (entrées)": "layers.0.B_re",
             "C (sorties)": "layers.0.C_re", "D (direct)": "layers.0.D"}
        return {k: v in names for k, v in m.items()}
    return {"g (amplitudes)": True, "W_in (entrées)": True}


def half_life(lam):
    a = np.clip(np.abs(lam), 1e-6, 0.99999)
    return np.log(0.5) / np.log(a)


def main():
    u, y = tasks.load(TASK)
    titles = []
    Win, W = TM.make_reservoir(N, rho=core.RHO, in_scale=core.IN_SCALE, sparsity=0.2, seed=SEED)
    fig, axes = plt.subplots(len(ROWS), 4, figsize=(21, 5.0 * len(ROWS) + 1),
                             gridspec_kw={"width_ratios": [1, 1.1, 1.1, 2.0], "hspace": 0.75, "wspace": 0.45})
    for i, (name, formula, kw, col) in enumerate(ROWS):
        kind = kw["kind"]
        m0 = TM.build_model(Win, W, kind, SEED, torch.float64, "cpu")
        lam0, g0 = eig(m0, kind), groups(m0, kind)
        s, info = TM.train_regression(u, y, Win, W, SEED, kind=kind, dtype=torch.float64, return_model=True)
        m = info["model"]
        lam1, g1 = eig(m, kind), groups(m, kind)
        print(f"{name}: NMSE {s:.3g}, {info['n_params']} paramètres", flush=True)

        ax = axes[i, 0]                                           # 1. valeurs propres
        t = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(t), np.sin(t), color="black", lw=0.6)
        ax.scatter(lam0.real, lam0.imag, s=18, color="lightgray", zorder=2, label="avant")
        ax.scatter(lam1.real, lam1.imag, s=22, color=col, zorder=3, label="après")
        if kind in ("lru", "linear_tuned"):                       # même ordre des modes avant et après : flèches
            for a, b in zip(lam0, lam1):
                if abs(b - a) > 0.01:
                    ax.annotate("", xy=(b.real, b.imag), xytext=(a.real, a.imag),
                                arrowprops=dict(arrowstyle="->", color=col, lw=0.8, alpha=0.7))
        ax.set_aspect("equal"); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.set_title("valeurs propres (cercle unité)", fontsize=9); ax.legend(fontsize=7, loc="lower left")
        npar = f"{info['n_params']:,}".replace(",", " ")
        titles.append((i, f"{name} — NMSE {s:.3g}, {npar} paramètres appris", formula))

        ax = axes[i, 1]                                           # 2. mémoire des modes
        h0, h1 = np.sort(half_life(lam0))[::-1], np.sort(half_life(lam1))[::-1]
        ax.plot(h0, color="lightgray", marker="o", ms=3, label="avant")
        ax.plot(h1, color=col, marker="o", ms=3, label="après")
        ax.set_yscale("log"); ax.set_xlabel("mode (du plus lent au plus rapide)", fontsize=8)
        if max(h0.max(), h1.max()) > 300:                       # |λ| ≥ 1 : pas de demi-vie (la tanh borne l'état)
            ax.set_ylim(top=300)
            ax.text(0.98, 0.97, "|λ| ≥ 1 coupé à 300 pas", transform=ax.transAxes, ha="right", va="top", fontsize=7)
        ax.set_ylabel("demi-vie (pas)", fontsize=8); ax.set_title("mémoire de chaque mode", fontsize=9)
        ax.axhline(10, color="black", lw=0.5, ls=":"); ax.text(len(h0) * 0.6, 11, "10 pas (ordre de NARMA10)", fontsize=7)
        ax.legend(fontsize=7)

        ax = axes[i, 2]                                           # 3. ce qui a bougé
        tr = trainable(m, kind)
        keys = list(g0)
        rel = [np.linalg.norm(g1[k] - g0[k]) / max(np.linalg.norm(g0[k]), 1e-12) for k in keys]
        ax.barh(range(len(keys)), rel, color=[col if tr[k] else "lightgray" for k in keys])
        for j, k in enumerate(keys):
            ax.text(rel[j], j, "  figé" if not tr[k] else f"  {rel[j]:.2f}", va="center", fontsize=7.5)
        ax.set_yticks(range(len(keys))); ax.set_yticklabels(keys, fontsize=8); ax.invert_yaxis()
        ax.set_xlabel("‖θ − θ₀‖ / ‖θ₀‖", fontsize=8); ax.set_title("ce qui a bougé pendant l'entraînement", fontsize=9)
        ax.set_xlim(0, max(rel) * 1.35 + 1e-3)
        ax.locator_params(axis="x", nbins=4)

        ax = axes[i, 3]                                           # 4. prédiction
        ut, yt = torch.as_tensor(u, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64)
        with torch.no_grad():
            X = m(ut.view(1, -1, 1))[0]
            w = TM.ridge(X[core.WASHOUT:core.n_train(u)], yt[core.WASHOUT:core.n_train(u)])
            p = (torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype)], 1) @ w).numpy()
        x = np.arange(T0, T0 + LEN)
        ax.plot(x, y[T0:T0 + LEN], color="black", lw=1.2, label="cible")
        ax.plot(x, p[T0:T0 + LEN], color=col, lw=1.2, ls="--", label="prédiction")
        ax.set_title(f"{TASK}, 120 pas de la série de test", fontsize=9); ax.legend(fontsize=7)
        ax.set_xlabel("pas", fontsize=8)
    for i, title, formula in titles:                           # titre de chaque ligne, au-dessus de ses panneaux
        pos = axes[i, 0].get_position()
        fig.text(pos.x0, pos.y1 + 0.045, title, fontsize=12, fontweight="bold")
        fig.text(pos.x0, pos.y1 + 0.028, formula, fontsize=9, color="dimgray")
    fig.suptitle(f"Partie 5.4 — récurrences linéaires accordées : ce que l'entraînement change ({N} neurones, "
                 f"{TASK})", fontsize=14, y=1.0)
    fig.savefig(core.out_path("figures", "linear_gallery.png"), dpi=105, bbox_inches="tight")
    print("Figure : results/figures/linear_gallery.png")


if __name__ == "__main__":
    main()
