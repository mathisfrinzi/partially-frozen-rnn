"""
Galerie des modèles : ce que chaque modèle fait au réservoir, sur un petit réseau qu'on peut dessiner.

Même réservoir de départ W₀ pour tous (20 neurones, 20 % de connexions), même tâche (NARMA10),
vrais entraînements avec le code de l'étude (1 500 pas). Pour chaque modèle :

  - à gauche, le réseau après entraînement (le score n'est pas affiché : à 20 neurones, les écarts
    entre modèles ne reflètent pas ceux de l'étude, qui grandissent avec la taille du réseau). Chaque trait est une connexion, colorée selon ce que
    l'entraînement lui a fait par rapport à W₀ : inchangée (gris), renforcée (rouge), affaiblie (bleu),
    créée (vert), coupée (pointillés). L'épaisseur et l'opacité suivent l'ampleur du changement.
  - à droite, ce qui est réellement appris, sous sa forme propre (ΔW, masque, rotation Q, gains g...).

    python experiments/11_model_gallery.py      (~1 min)   -> results/figures/model_gallery.png
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import matplotlib
import matplotlib.colors
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from multiprocessing import Pool

from hfrnn import core, tasks
from hfrnn import params as P

N, DENSITY, SEED, TASK = 20, 0.2, 3, "NARMA10"

# (nom, formule, paramétrisation, ce qu'on montre à droite)
MODELS = [
    ("Réservoir classique", "W = W₀", None, "W0"),
    ("RNN classique", "W apprise en entier", lambda: P.Convex(1.0), "dW"),
    ("RNN freiné", "récurrence 10× plus lente", lambda: P.Convex(1.0, lr_mult=0.1, lr_mult_in=1.0), "dW"),
    ("Mélange fixe (α = 0.1)", "W = 0.9·W₀ + 0.1·ΔW", lambda: P.Convex(0.1), "dW"),
    ("Mélange réglé", "W = (1−α)·W₀ + α·ΔW, α appris", lambda: P.LearnedAlpha(0.1), "dW"),
    ("Réservoir corrigé de rang 2", "W = W₀ + U·Vᵀ", lambda: P.LowRank(2), "dW"),
    ("Réservoir retouché", "W = W₀ + M ⊙ ΔW", lambda: P.SparseMask(), "dW"),
    ("Réservoir sculpté intégral", "W = m ⊙ W₀,  m ∈ [0, 1]",
     lambda: P.SuperMask("sigmoid", win="mask", lr_mult=50.0), "mask"),
    ("Réservoir élagué (50 %)", "W = m ⊙ W₀,  m ∈ {0, 1}",
     lambda: P.SuperMask("topk", keep=0.5, win="free", lr_mult=10.0, rescale=True), "mask"),
    ("Réservoir repondéré (contrôle)", "W = g ⊙ W₀,  g libre",
     lambda: P.SuperMask("gain", win="mask", lr_mult=2.5), "gain"),
    ("Réservoir tourné", "W = Q·W₀·Qᵀ,  Q orthogonale", lambda: P.Rotation("rotate"), "Q"),
    ("Réservoir accordé", "W = U·diag(σ ⊙ g)·Vᵀ", lambda: P.SpectrumOnly(), "g"),
]

C = {"inchangée": "#b0b0b0", "renforcée": "#d62728", "affaiblie": "#1f77b4", "créée": "#2ca02c",
     "coupée": "#7f7f7f"}


def reservoir():
    return core.make_reservoir(np.random.default_rng(SEED), n=N, sparsity=DENSITY)


def fit(i):
    """Entraîne le modèle i ; renvoie la NMSE, les matrices finales et l'objet appris."""
    name, _, make, _ = MODELS[i]
    u, y = tasks.load(TASK)
    Win0, W0 = reservoir()
    if make is None:
        return {"nmse": P.pure_reservoir(u, y, Win0, W0), "Win": Win0, "W": W0}
    p = make()
    score, info = P.train(p, u, y, Win0, W0, SEED)
    Win, W = p.compose(Win0, W0)
    out = {"nmse": score, "Win": Win, "W": W, "info": info}
    if isinstance(p, P.SuperMask):
        out["m"] = p.mask()                               # sans le facteur de remise à l'échelle
    elif isinstance(p, P.Rotation):
        out["Q"] = p.Q()
    elif isinstance(p, P.SpectrumOnly):
        out["g"], out["sig"] = p.g.copy(), p.sig.copy()
    return out


def draw_network(ax, Win0, W0, Win, W, scale):
    ang = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.pi / 2
    pos = np.c_[np.cos(ang), np.sin(ang)]
    tol = 0.05 * scale                                   # changements plus petits : ignorés
    segs = {k: ([], [], []) for k in C}
    for i in range(N):
        for j in range(N):
            a, b = W0[i, j], W[i, j]
            if a == 0 and abs(b) < tol:
                continue
            if a != 0 and abs(b) < tol:
                kind = "coupée"
            elif a == 0:
                kind = "créée"
            elif abs(b - a) < tol:
                kind = "inchangée"
            else:
                kind = "renforcée" if abs(b) > abs(a) else "affaiblie"
            p0, p1 = pos[j], pos[i]
            if i == j:                                       # boucle sur soi : petit cercle
                t = np.linspace(0, 2 * np.pi, 20)
                c = p0 * 1.12
                seg = np.c_[c[0] + 0.1 * np.cos(t), c[1] + 0.1 * np.sin(t)]
            else:
                mid = (p0 + p1) / 2 * 0.75                   # courbe légère vers le centre
                t = np.linspace(0, 1, 12)[:, None]
                seg = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * mid + t ** 2 * p1
            amp = min(1, abs(b - a) / (0.5 * scale))         # ampleur du changement, 0 à 1
            if kind == "inchangée":
                w, al = 0.7, 0.8
            elif kind == "coupée":
                w, al = 0.9, 0.7
            else:
                w, al = 0.5 + 3.5 * amp, 0.15 + 0.8 * amp
            segs[kind][0].append(seg); segs[kind][1].append(w); segs[kind][2].append(al)
    for kind in ("inchangée", "coupée", "créée", "affaiblie", "renforcée"):
        s, w, al = segs[kind]
        if s:
            rgba = [matplotlib.colors.to_rgba(C[kind], a) for a in al]
            ax.add_collection(LineCollection(s, colors=rgba, linewidths=w,
                                             linestyles=(0, (1, 2)) if kind == "coupée" else "-"))
    # entrée
    inp = np.array([-1.75, 0.0])
    din = np.abs(Win - Win0).ravel()
    for i in range(N):
        amp = min(1, din[i] / 0.5)
        col = "#b0b0b0" if din[i] < 0.025 else ("#d62728" if abs(Win[i, 0]) > abs(Win0[i, 0]) else "#1f77b4")
        ax.plot(*np.c_[inp, pos[i]], color=col, lw=0.5 + 2 * amp, alpha=0.6 if col == "#b0b0b0" else 0.2 + 0.7 * amp,
                zorder=1)
    ax.scatter(*pos.T, s=40, c="black", zorder=3)
    ax.scatter(*inp, s=110, marker="s", c="tab:green", zorder=3)
    ax.text(inp[0], inp[1] + 0.18, "u(t)", ha="center", fontsize=8)
    ax.set_xlim(-1.95, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal"); ax.axis("off")


def draw_learned(ax, kind, r, W0, dscale):
    ax.tick_params(labelsize=6)
    if kind == "W0":
        im = ax.imshow(W0, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_title("W₀ (figée)\nseul le readout est appris", fontsize=8)
    elif kind == "dW":
        im = ax.imshow(r["W"] - W0, cmap="PiYG", vmin=-dscale, vmax=dscale)
        t = "changement W − W₀"
        if "alpha_rec" in r.get("info", {}):
            t += f"\nα_in → {r['info']['alpha_in']:.2f},  α_rec → {r['info']['alpha_rec']:.2f}"
        ax.set_title(t, fontsize=8)
    elif kind == "mask":
        m = np.where(W0 != 0, r["m"], np.nan)
        im = ax.imshow(m, cmap="Purples", vmin=0, vmax=1)
        ax.set_title("masque m appris\n(blanc : pas de connexion)", fontsize=8)
    elif kind == "gain":
        g = np.where(W0 != 0, r["m"], np.nan)
        lim = max(1.5, np.nanmax(np.abs(g - 1)) + 1)
        im = ax.imshow(g, cmap="RdBu_r", vmin=2 - lim, vmax=lim)
        ax.set_title("gain g appris\n(< 0 : connexion inversée)", fontsize=8)
    elif kind == "Q":
        im = ax.imshow(r["Q"] - np.eye(N), cmap="PiYG", vmin=-0.5, vmax=0.5)
        ax.set_title("rotation apprise Q − I\n(spectre de W₀ inchangé)", fontsize=8)
    elif kind == "g":
        ax.bar(range(N), r["sig"], color="lightgray", label="σ de W₀")
        ax.bar(range(N), r["sig"] * r["g"], color="tab:blue", width=0.5, label="σ ⊙ g appris")
        ax.set_title("amplitudes par direction\n(directions de W₀ inchangées)", fontsize=8)
        ax.set_xlabel("direction", fontsize=7); ax.legend(fontsize=6); ax.set_box_aspect(1)
        return
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03).ax.tick_params(labelsize=6)


def main():
    with Pool(min(len(MODELS), os.cpu_count() or 1)) as pool:
        res = pool.map(fit, range(len(MODELS)))
    Win0, W0 = reservoir()
    scale = np.abs(W0).max()
    dscale = max(np.abs(r["W"] - W0).max() for r, m in zip(res, MODELS) if m[3] == "dW")

    ncol = 3
    nrow = int(np.ceil(len(MODELS) / ncol))
    fig = plt.figure(figsize=(21, 3.5 * nrow + 1.8))
    gs = fig.add_gridspec(nrow, ncol * 2, width_ratios=[1.7, 1] * ncol, hspace=0.38, wspace=0.08,
                          top=1 - 1.9 / (3.5 * nrow + 1.8), bottom=0.02)
    for k, ((name, formula, _, kind), r) in enumerate(zip(MODELS, res)):
        row, col = divmod(k, ncol)
        axn = fig.add_subplot(gs[row, 2 * col]); axl = fig.add_subplot(gs[row, 2 * col + 1])
        draw_network(axn, Win0, W0, r["Win"], r["W"], scale)
        axn.set_title(f"{name}\n{formula}", fontsize=11, fontweight="bold", loc="left")
        draw_learned(axl, kind, r, W0, dscale)
    handles = [Line2D([], [], color=C[k], lw=2.5, ls=":" if k == "coupée" else "-", label=k) for k in C]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, 1 - 0.45 / (3.5 * nrow + 1.8)), title_fontsize=11,
               title="Connexions comparées au réservoir de départ W₀ (épaisseur et opacité = ampleur du "
                     "changement ; changements de moins de 5 % ignorés)")
    fig.suptitle(f"Ce que chaque modèle fait au réservoir — même W₀ ({N} neurones), même tâche ({TASK})",
                 fontsize=15, y=1.0)
    fig.savefig(core.out_path("figures", "model_gallery.png"), dpi=110, bbox_inches="tight")
    print("Figure : results/figures/model_gallery.png")


if __name__ == "__main__":
    main()
