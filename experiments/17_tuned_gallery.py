"""
Galerie de la partie 4.2 : ce que chaque variante du réservoir accordé apprend, sur un petit réseau.

Même réservoir de départ W₀ (20 neurones, 20 % de connexions), même tâche (NARMA10), vrais entraînements
(version PyTorch, 1 500 pas). Pour chaque variante :
  - à gauche, le réseau après entraînement, connexions colorées comme dans la galerie de la partie 1
    (inchangée, renforcée, affaiblie, créée, coupée) ;
  - à droite, ce qui est appris, sous sa forme propre (amplitudes, vitesses, valeurs propres, coefficients...).

    python experiments/17_tuned_gallery.py       (~3 min)   -> results/figures/tuned_gallery.png
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import torch

from hfrnn import core, tasks
from hfrnn import torch_models as TM

N, DENSITY, SEED, TASK = 20, 0.2, 3, "NARMA10"
C = {"inchangée": "#b0b0b0", "renforcée": "#d62728", "affaiblie": "#1f77b4", "créée": "#2ca02c",
     "coupée": "#7f7f7f"}
VARIANTS = [
    ("Réservoir accordé (partie 3)", "W = U·diag(σ ⊙ g)·Vᵀ", {"kind": "tuned"}, "g"),
    ("Accordé borné", "g ∈ [0, 2]", {"kind": "tuned", "gmax": 2.0}, "g"),
    ("Accordé + fuite", "x ← (1 − a) ⊙ x + a ⊙ tanh(…)", {"kind": "tuned", "leak": True}, "leak"),
    ("Accordé, valeurs propres", "W = P·diag(λ ⊙ g ⊙ e^{iφ})·P⁻¹", {"kind": "tuned", "basis": "eig"}, "eig"),
    ("Accordé paramétrique", "g = 1 + Σₖ cₖ Tₖ(z), 8 coefficients", {"kind": "tuned", "poly": 8}, "poly"),
    ("Accordé + rang 4", "W = U·diag(σ ⊙ g)·Vᵀ + A·Bᵀ", {"kind": "tuned", "rank": 4}, "rank"),
    ("Accordé profond", "deux réservoirs accordés de N/2 empilés", {"kind": "deep_tuned", "sparsity": DENSITY}, "deep"),
]


def circle(n, center=(0.0, 0.0), r=1.0):
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
    return np.c_[center[0] + r * np.cos(ang), center[1] + r * np.sin(ang)]


def edges(ax, pos, W0, W, scale, inner=0.75):
    """Connexions colorées selon le changement par rapport à W₀ (comme 11_model_gallery.py)."""
    n = len(pos)
    tol = 0.05 * scale
    center = pos.mean(0)
    segs = {k: ([], [], []) for k in C}
    for i in range(n):
        for j in range(n):
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
            if i == j:
                t = np.linspace(0, 2 * np.pi, 20)
                c = center + (p0 - center) * 1.12
                seg = np.c_[c[0] + 0.1 * np.cos(t), c[1] + 0.1 * np.sin(t)]
            else:
                mid = center + ((p0 + p1) / 2 - center) * inner
                t = np.linspace(0, 1, 12)[:, None]
                seg = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * mid + t ** 2 * p1
            amp = min(1, abs(b - a) / (0.5 * scale))
            w, al = (0.7, 0.8) if kind == "inchangée" else (0.9, 0.7) if kind == "coupée" else (0.5 + 3.5 * amp,
                                                                                                 0.15 + 0.8 * amp)
            segs[kind][0].append(seg); segs[kind][1].append(w); segs[kind][2].append(al)
    for kind in ("inchangée", "coupée", "créée", "affaiblie", "renforcée"):
        s, w, al = segs[kind]
        if s:
            ax.add_collection(LineCollection(s, colors=[matplotlib.colors.to_rgba(C[kind], x) for x in al],
                                             linewidths=w, linestyles=(0, (1, 2)) if kind == "coupée" else "-"))


def inputs(ax, src, pos, Win0, Win):
    din = np.abs(Win - Win0).sum(1)
    for i in range(len(pos)):
        amp = min(1, din[i] / 0.5)
        col = "#b0b0b0" if din[i] < 0.025 else ("#d62728" if np.abs(Win[i]).sum() > np.abs(Win0[i]).sum() else "#1f77b4")
        ax.plot(*np.c_[src, pos[i]], color=col, lw=0.5 + 2 * amp, alpha=0.6 if col == "#b0b0b0" else 0.2 + 0.7 * amp,
                zorder=1)


def train(kw):
    u, y = tasks.load(TASK)
    kw = dict(kw)
    sp = kw.pop("sparsity", None)
    Win0, W0 = TM.make_reservoir(N, rho=core.RHO, in_scale=core.IN_SCALE, sparsity=DENSITY, seed=SEED)
    if sp is not None:
        kw["sparsity"] = sp
    _, info = TM.train_regression(u, y, Win0, W0, SEED, dtype=torch.float64, return_model=True, **kw)
    return Win0, W0, info["model"]


def numpy(t):
    return t.detach().cpu().numpy()


def draw_single(axn, axl, kind, Win0, W0, m):
    Win, W = (numpy(x) for x in m.weights())
    scale = np.abs(W0).max()
    pos = circle(N)
    edges(axn, pos, W0, W, scale)
    inputs(axn, np.array([-1.75, 0.0]), pos, Win0, Win)
    node_c = "black"
    if kind == "leak":
        a = numpy(torch.sigmoid(m.c_leak))
        lo = min(0.85, a.min() - 0.01)
        axn.scatter(*pos.T, s=80, c=a, cmap="plasma", vmin=lo, vmax=1, zorder=3, edgecolors="black")
        node_c = None
    if node_c:
        axn.scatter(*pos.T, s=40, c=node_c, zorder=3)
    axn.scatter(-1.75, 0, s=110, marker="s", c="tab:green", zorder=3)
    axn.text(-1.75, 0.18, "u(t)", ha="center", fontsize=8)
    axn.set_xlim(-1.95, 1.3); axn.set_ylim(-1.3, 1.3); axn.set_aspect("equal"); axn.axis("off")

    axl.tick_params(labelsize=6)
    if kind in ("g", "rank", "poly"):
        sig = numpy(m.sig); g = numpy(m.gains())
        axl.bar(range(N), sig, color="lightgray", label="σ de W₀")
        axl.bar(range(N), sig * g, width=0.5, color="tab:blue", label="σ ⊙ g appris")
        if m.gmax:
            axl.plot(range(N), m.gmax * sig, color="tab:red", ls=":", lw=1, label="borne 2σ")
        if kind == "poly":
            ax2 = axl.twinx(); ax2.plot(range(N), g, color="tab:orange", lw=2, label="g (8 coefficients)")
            ax2.set_ylabel("g", fontsize=7, color="tab:orange"); ax2.tick_params(labelsize=6)
        axl.set_xlabel("direction (σ décroissante)", fontsize=7)
        if kind != "rank":
            axl.legend(fontsize=6, loc="upper right")
        axl.set_title("amplitudes par direction", fontsize=8)
        if kind == "rank":
            ins = axl.inset_axes([0.45, 0.47, 0.4, 0.4])
            AB = numpy(m.A @ m.B.T); lim = np.abs(AB).max()
            ins.imshow(AB, cmap="PiYG", vmin=-lim, vmax=lim); ins.set_xticks([]); ins.set_yticks([])
            ins.set_title("correction A·Bᵀ (rang 4)", fontsize=6)
    elif kind == "leak":
        a = numpy(torch.sigmoid(m.c_leak))
        lo = min(0.85, a.min() - 0.01)
        axl.bar(range(N), a, color=plt.cm.plasma((a - lo) / (1 - lo)))
        axl.axhline(1, color="gray", ls=":", lw=1)
        axl.set_ylim(lo, 1.005); axl.set_xlabel("neurone (couleur = couleur du nœud)", fontsize=7)
        axl.set_title("vitesse a de chaque neurone\n(1 = sans mémoire propre, < 1 = plus lent)", fontsize=8)
    elif kind == "eig":
        lam0 = numpy(m.lam)
        g = numpy(m.g); ph = numpy(m.phi); pair = numpy(m.pair)
        ge, phe = (g + g[pair]) / 2, (ph - ph[pair]) / 2
        lam1 = lam0 * ge * np.exp(1j * phe)
        t = np.linspace(0, 2 * np.pi, 200)
        axl.plot(np.cos(t), np.sin(t), color="lightgray", lw=1)
        axl.scatter(lam0.real, lam0.imag, s=18, c="lightgray", edgecolors="gray", label="λ de W₀", zorder=2)
        axl.scatter(lam1.real, lam1.imag, s=18, c="tab:blue", label="λ appris", zorder=3)
        for a0, a1 in zip(lam0, lam1):
            axl.annotate("", xy=(a1.real, a1.imag), xytext=(a0.real, a0.imag),
                         arrowprops=dict(arrowstyle="->", color="tab:blue", lw=0.6, alpha=0.6))
        axl.set_aspect("equal"); axl.legend(fontsize=6, loc="lower left")
        axl.set_title("valeurs propres (plan complexe)\nmodule et phase de chaque mode", fontsize=8)


def draw_deep(axn, axl, Win0, W0, m):
    h = N // 2
    s = 0.62
    pos1, pos2 = circle(h, (-0.55, 0), s), circle(h, (0.75, 0), s)
    for layer, pos in ((m.l1, pos1), (m.l2, pos2)):
        W0l = numpy(layer.W_r); Wl = numpy(layer.weights()[1])
        edges(axn, pos, W0l, Wl, np.abs(W0l).max())
        axn.scatter(*pos.T, s=30, c="black", zorder=3)
    inputs(axn, np.array([-1.75, 0.0]), pos1, numpy(m.l1.Win_r), numpy(m.l1.weights()[0]))
    # couche 1 -> couche 2 : quelques liaisons, en gris
    for i in range(h):
        for j in range(0, h, 3):
            axn.plot(*np.c_[pos1[j], pos2[i]], color="#b0b0b0", lw=0.3, alpha=0.4, zorder=1)
    axn.scatter(-1.75, 0, s=110, marker="s", c="tab:green", zorder=3)
    axn.text(-1.75, 0.18, "u(t)", ha="center", fontsize=8)
    axn.text(-0.55, -0.85, "réservoir 1", ha="center", fontsize=8); axn.text(0.75, -0.85, "réservoir 2", ha="center", fontsize=8)
    axn.set_xlim(-1.95, 1.5); axn.set_ylim(-1.3, 1.3); axn.set_aspect("equal"); axn.axis("off")
    for k, (layer, col) in enumerate(((m.l1, "tab:blue"), (m.l2, "tab:cyan"))):
        sig = numpy(layer.sig); g = numpy(layer.gains())
        x = np.arange(h) + k * (h + 1)
        axl.bar(x, sig, color="lightgray"); axl.bar(x, sig * g, width=0.5, color=col, label=f"réservoir {k + 1}")
    axl.legend(fontsize=6); axl.tick_params(labelsize=6)
    axl.set_title("amplitudes par direction,\nchaque réservoir", fontsize=8)


def main():
    ncol = 3
    nrow = int(np.ceil(len(VARIANTS) / ncol))
    H = 3.6 * nrow + 1.8
    fig = plt.figure(figsize=(21, H))
    gs = fig.add_gridspec(nrow, ncol * 2, width_ratios=[1.7, 1] * ncol, hspace=0.42, wspace=0.12,
                          top=1 - 1.9 / H, bottom=0.03)
    for k, (name, formula, kw, kind) in enumerate(VARIANTS):
        Win0, W0, m = train(kw)
        row, col = divmod(k, ncol)
        axn = fig.add_subplot(gs[row, 2 * col]); axl = fig.add_subplot(gs[row, 2 * col + 1])
        if kind == "deep":
            draw_deep(axn, axl, Win0, W0, m)
        else:
            draw_single(axn, axl, kind, Win0, W0, m)
        axn.set_title(f"{name}\n{formula}", fontsize=11, fontweight="bold", loc="left")
        print(name, "ok", flush=True)
    handles = [Line2D([], [], color=C[k], lw=2.5, ls=":" if k == "coupée" else "-", label=k) for k in C]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=11, frameon=False,
               bbox_to_anchor=(0.5, 1 - 0.45 / H), title_fontsize=11,
               title="Connexions comparées au réservoir de départ W₀ (épaisseur et opacité = ampleur du changement)")
    fig.suptitle(f"Partie 4.2 — variantes du réservoir accordé, même W₀ ({N} neurones), même tâche ({TASK})",
                 fontsize=15, y=1.0)
    fig.savefig(core.out_path("figures", "tuned_gallery.png"), dpi=110, bbox_inches="tight")
    print("Figure : results/figures/tuned_gallery.png")


if __name__ == "__main__":
    main()
