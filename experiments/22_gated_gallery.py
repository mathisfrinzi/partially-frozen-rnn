"""
Galerie de la partie 5 : le réservoir accordé à portes et ses variantes, une variante par ligne.

Même réservoir de départ W₀ (20 neurones, 20 % de connexions), même tâche (NARMA10), vrais entraînements
(version PyTorch, 1 500 pas). Pour chaque variante, trois panneaux :
  - le réseau après entraînement : connexions de la matrice principale colorées comme dans les autres galeries
    (inchangée, renforcée, affaiblie, créée, coupée) ; couleur des neurones = ouverture moyenne de la porte ;
  - les amplitudes apprises : celles de la matrice principale (σ ⊙ g) et celles des matrices des portes ;
  - la porte au fil du temps : ouverture de chaque neurone pendant 120 pas de la série de test, sous l'entrée u(t).

    python experiments/22_gated_gallery.py       (~5 min)   -> results/figures/gated_gallery.png
"""
import importlib.util
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import torch

from hfrnn import core, tasks

spec = importlib.util.spec_from_file_location("tg", Path(__file__).with_name("17_tuned_gallery.py"))
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)
N, TASK = G.N, G.TASK
T0, LEN, WARM = 3200, 120, 200          # fenêtre affichée (partie test) et pas de mise en route avant elle

VARIANTS = [
    ("Réservoir accordé à portes (type GRU)",
     "z, r = σ(W_z h + B u + c),   h̃ = tanh(W (r ⊙ h) + W_in u),   h ← (1 − z) ⊙ h + z ⊙ h̃   — W, W_z, W_r accordées",
     {"kind": "gated_tuned", "cell": "gru"}, "z"),
    ("Réservoir accordé à portes légères",
     "comme ci-dessus, mais chaque porte ne regarde que son propre neurone :  z = σ(a_z ⊙ h + B u + c)",
     {"kind": "gated_tuned", "cell": "gru_light"}, "z"),
    ("Réservoir accordé à portes, type LSTM",
     "f, i, o = σ(W_k h + B u + c),   c ← f ⊙ c + i ⊙ tanh(W h + W_in u),   h = o ⊙ tanh(c)   — W, W_f, W_i, W_o accordées",
     {"kind": "gated_tuned", "cell": "lstm"}, "f"),
    ("Réservoir accordé à porte d'entrée (type minGRU)",
     "z = σ(B u + c) ne dépend que de l'entrée,   h ← (1 − z) ⊙ h + z ⊙ tanh(W h + W_in u)   — seule W est accordée",
     {"kind": "gated_tuned", "cell": "mingru"}, "z"),
    ("Contrôle : portes seules",
     "même cellule que la première ligne, mais W reste le réservoir W₀ : seules les portes apprennent (esprit des Gated ESN)",
     {"kind": "gated_tuned", "cell": "gru", "gates_only": True}, "z"),
]
GATE_NAME = {"z": "z (mise à jour)", "r": "r (remise à zéro)", "f": "f (oubli)", "i": "i (entrée)", "o": "o (sortie)"}
GATE_COL = {"z": "tab:orange", "r": "tab:green", "f": "tab:purple", "i": "tab:orange", "o": "tab:green"}


def numpy(t):
    return t.detach().cpu().numpy()


@torch.no_grad()
def record(m, u):
    """Fait tourner la cellule pas à pas et garde les valeurs des portes."""
    h = torch.zeros(1, m.n_features, dtype=m.sig.dtype)
    c = torch.zeros_like(h)
    Win = m.Win_r + m.Win_l
    out = {k: [] for k in m.GATES[m.cell]}
    for t in range(len(u)):
        ut = torch.tensor([[u[t]]], dtype=h.dtype)
        zdir = h @ m.Vs
        inp = ut @ Win.T + m.bias
        g = {k: m._gate(k, h, zdir, ut) for k in m.GATES[m.cell]}
        if m.cell in ("gru", "gru_light"):
            cand = torch.tanh(m._rec((g["r"] * h) @ m.Vs, m.g) + inp)
            h = (1 - g["z"]) * h + g["z"] * cand
        elif m.cell == "mingru":
            h = (1 - g["z"]) * h + g["z"] * torch.tanh(m._rec(zdir, m.g) + inp)
        else:
            c = g["f"] * c + g["i"] * torch.tanh(m._rec(zdir, m.g) + inp)
            h = g["o"] * torch.tanh(c)
        for k in out:
            out[k].append(numpy(g[k])[0])
    return {k: np.array(v) for k, v in out.items()}


def main_matrix(m):
    return numpy((m.Us * (m.sig * m.g)) @ m.Vs.T)


def draw(fig, gs_row, name, formula, kind_gate, Win0, W0, m, u):
    sub = gs_row.subgridspec(2, 3, width_ratios=[1.0, 1.35, 2.4], height_ratios=[1, 4.2], wspace=0.22, hspace=0.06)
    axn = fig.add_subplot(sub[:, 0]); axl = fig.add_subplot(sub[:, 1])
    axu = fig.add_subplot(sub[0, 2]); axg = fig.add_subplot(sub[1, 2], sharex=axu)

    gates = record(m, u[T0 - WARM:T0 + LEN])
    gates = {k: v[WARM:] for k, v in gates.items()}
    main = gates[kind_gate]
    mean = main.mean(0)

    # 1. réseau
    W = main_matrix(m)
    pos = G.circle(N)
    G.edges(axn, pos, W0, W, np.abs(W0).max())
    G.inputs(axn, np.array([-1.75, 0.0]), pos, Win0, numpy(m.Win_r + m.Win_l))
    sc = axn.scatter(*pos.T, s=90, c=mean, cmap="viridis", vmin=0, vmax=1, zorder=3, edgecolors="black", lw=0.6)
    axn.scatter(-1.75, 0, s=110, marker="s", c="tab:green", zorder=3)
    axn.text(-1.75, 0.18, "u(t)", ha="center", fontsize=8)
    axn.set_xlim(-1.95, 1.3); axn.set_ylim(-1.3, 1.3); axn.set_aspect("equal"); axn.axis("off")
    axn.text(0, -1.42, f"couleur des neurones : {GATE_NAME[kind_gate]} moyenne", ha="center", fontsize=7.5)

    # 2. amplitudes apprises
    x = np.arange(N)
    sig, g = numpy(m.sig), numpy(m.g)
    axl.bar(x, sig, color="lightgray", label="σ de W₀")
    axl.bar(x, sig * g, width=0.5, color="tab:blue", label="W : σ ⊙ g")
    if m.cell in ("gru", "lstm"):
        for k in m.GATES[m.cell]:
            axl.plot(x, sig * numpy(getattr(m, f"g_{k}")), marker="o", ms=3, lw=1.2, color=GATE_COL[k],
                     label=f"W_{k} : σ ⊙ g_{k}")
        axl.set_xlabel("direction (σ décroissante)", fontsize=7)
        axl.set_title("amplitudes par direction\n(les portes partent de 0)", fontsize=8)
    elif m.cell == "gru_light":
        for k in ("z", "r"):
            axl.plot(x, numpy(getattr(m, f"a_{k}")), marker="o", ms=3, lw=1.2, color=GATE_COL[k],
                     label=f"a_{k} (par neurone)")
        axl.set_xlabel("direction de W (barres) / neurone (courbes)", fontsize=7)
        axl.set_title("amplitudes de W et poids\nde chaque porte sur son neurone", fontsize=8)
    else:
        axl.plot(x, numpy(m.B_z)[:, 0], marker="o", ms=3, lw=1.2, color=GATE_COL["z"], label="B_z (poids de u)")
        axl.set_xlabel("direction de W (barres) / neurone (courbe)", fontsize=7)
        axl.set_title("amplitudes de W et poids\nde l'entrée sur la porte", fontsize=8)
    axl.set_xticks(range(0, N, 5))
    axl.axhline(0, color="black", lw=0.5)
    axl.legend(fontsize=6, loc="upper right"); axl.tick_params(labelsize=6)

    # 3. la porte au fil du temps
    t = np.arange(LEN)
    axu.plot(t, u[T0:T0 + LEN], color="tab:green", lw=1)
    axu.set_ylabel("u(t)", fontsize=7); axu.tick_params(labelsize=6, labelbottom=False)
    axu.set_title(f"ouverture de la porte {GATE_NAME[kind_gate]}, neurone par neurone, pendant {LEN} pas de test",
                  fontsize=8)
    lo, hi = float(main.min()), float(main.max())
    im = axg.imshow(main.T, aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    axg.set_yticks(range(0, N, 5))
    axg.set_ylabel("neurone", fontsize=7); axg.set_xlabel("pas de temps", fontsize=7); axg.tick_params(labelsize=6)
    others = "   ".join(f"{k} moyen = {v.mean():.2f}" for k, v in gates.items())
    axg.text(1.0, -0.32, f"plage de {kind_gate} : {lo:.2f} – {hi:.2f}     {others}", transform=axg.transAxes,
             ha="right", fontsize=7, color="dimgray")
    cb = fig.colorbar(im, ax=axg, fraction=0.025, pad=0.01); cb.ax.tick_params(labelsize=6)

    n_rec = sum(p.numel() for p in m.recurrent_params())
    box = gs_row.get_position(fig)
    fig.text(box.x0, box.y1 + 0.75 / fig.get_figheight(), f"{name}   ({n_rec} paramètres appris hors lecture, N = {N})",
             fontsize=12.5, fontweight="bold", ha="left", va="bottom")
    fig.text(box.x0, box.y1 + 0.47 / fig.get_figheight(), formula, fontsize=9.5, color="dimgray", ha="left", va="bottom")


def main():
    u, _ = tasks.load(TASK)
    nrow = len(VARIANTS)
    H = 4.3 * nrow + 1.6
    fig = plt.figure(figsize=(20, H))
    gs = fig.add_gridspec(nrow, 1, hspace=0.85, top=1 - 2.1 / H, bottom=0.03, left=0.03, right=0.97)
    for k, (name, formula, kw, gate) in enumerate(VARIANTS):
        Win0, W0, m = G.train(kw)
        draw(fig, gs[k], name, formula, gate, Win0, W0, m, u)
        print(name, "ok", flush=True)
    handles = [Line2D([], [], color=G.C[k], lw=2.5, ls=":" if k == "coupée" else "-", label=k) for k in G.C]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=10.5, frameon=False,
               bbox_to_anchor=(0.5, 1 - 0.42 / H),
               title="Connexions de la matrice principale W comparées au réservoir de départ W₀", title_fontsize=10.5)
    fig.suptitle(f"Partie 5 — réservoir accordé à portes et variantes, même W₀ ({N} neurones), même tâche ({TASK})",
                 fontsize=15, y=1.0)
    fig.savefig(core.out_path("figures", "gated_gallery.png"), dpi=110, bbox_inches="tight")
    print("Figure : results/figures/gated_gallery.png")


if __name__ == "__main__":
    main()
