"""
Figure de présentation : réservoir classique, RNN classique, réservoir accordé, côte à côte.

Pour chaque modèle : le réseau après entraînement (20 neurones, NARMA10, comme la galerie), ce qui est appris,
les équations, le nombre de paramètres récurrents appris et les résultats à 500 neurones (GPU, 3 seeds).

    python experiments/20_pitch_figure.py      (~1 min)   -> results/figures/pitch.png
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

from hfrnn.core import out_path

spec = importlib.util.spec_from_file_location("gallery", Path(__file__).with_name("11_model_gallery.py"))
G = importlib.util.module_from_spec(spec)
spec.loader.exec_module(G)

INK, MUTED, ACCENT = "#1f2328", "#57606a", "#0b5cad"
COLUMNS = [
    {"name": "Réservoir classique", "idx": 0, "kind": "W0", "color": "#6e7781",
     "eq": ["x(t+1) = tanh( W₀·x(t) + W_in·u(t) + b )", "ŷ(t) = W_out·x(t)"],
     "learned": "appris : W_out seulement", "params": "0", "label": "réservoir classique"},
    {"name": "RNN classique", "idx": 1, "kind": "dW", "color": "#cf222e",
     "eq": ["x(t+1) = tanh( W·x(t) + W_in·u(t) + b )", "ŷ(t) = W_out·x(t)"],
     "learned": "appris : W (N² poids), W_in, b, W_out", "params": "N²  (250 000)", "label": "RNN"},
    {"name": "Réservoir accordé", "idx": 11, "kind": "g", "color": ACCENT,
     "eq": ["x(t+1) = tanh( U·diag(σ ⊙ g)·Vᵀ·x(t) + W_in·u(t) + b )", "ŷ(t) = W_out·x(t)",
            "W₀ = U·diag(σ)·Vᵀ : directions U, V aléatoires figées"],
     "learned": "appris : g (une amplitude par direction), W_in, b, W_out", "params": "N  (500)",
     "label": "réservoir accordé"},
]


def results_500():
    """NMSE à 500 neurones (GPU, 3 seeds) : réservoir classique, meilleur RNN, réservoir accordé."""
    R = json.load(open(out_path("data", "gpu_regression_n500.json")))["results"]
    out = {}
    for t in ("Mackey-Glass (+15)", "NARMA10"):
        r = R[t]
        rnn = min((l for l in r if l == "RNN" or l.startswith("RNN, lr global")), key=lambda l: np.mean(np.log10(r[l])))
        out[t] = {"réservoir classique": np.mean(r["réservoir classique"]), "RNN": np.mean(r[rnn]),
                  "réservoir accordé": np.mean(r["réservoir accordé"])}
    return out


def main():
    res = results_500()
    fits = {c["idx"]: G.fit(c["idx"]) for c in COLUMNS}
    Win0, W0 = G.reservoir()
    scale = np.abs(W0).max()
    dscale = np.abs(fits[1]["W"] - W0).max()

    fig = plt.figure(figsize=(21, 10.4), facecolor="white")
    gs = fig.add_gridspec(3, 6, width_ratios=[1.6, 1] * 3, height_ratios=[3.6, 1.9, 1.5],
                          left=0.02, right=0.98, top=0.84, bottom=0.03, hspace=0.06, wspace=0.12)
    fig.text(0.5, 0.965, "Apprendre peu, à partir du hasard", ha="center", fontsize=24, fontweight="bold", color=INK)
    fig.text(0.5, 0.925, "Un réservoir aléatoire, dont on n'apprend que l'amplitude de chaque direction, "
             "bat un réseau récurrent entièrement appris.", ha="center", fontsize=13.5, color=MUTED)
    handles = [Line2D([], [], color=G.C[k], lw=2.5, ls=":" if k == "coupée" else "-", label=k) for k in G.C]
    fig.legend(handles=handles, loc="upper center", ncol=5, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, 0.905),
               title="connexions comparées au réservoir de départ W₀ (épaisseur = ampleur du changement)",
               title_fontsize=10)

    for j, c in enumerate(COLUMNS):
        r = fits[c["idx"]]
        axn = fig.add_subplot(gs[0, 2 * j]); axl = fig.add_subplot(gs[0, 2 * j + 1])
        G.draw_network(axn, Win0, W0, r["Win"], r["W"], scale)
        G.draw_learned(axl, c["kind"], r, W0, dscale)
        if c["kind"] == "g":
            axl.set_xticks([0, 5, 10, 15, 19])
        axn.set_title(c["name"], fontsize=17, fontweight="bold", loc="left", color=c["color"], pad=10)

        # équations et ce qui est appris
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2]); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.01, 0.04), 0.98, 0.92, boxstyle="round,pad=0.01,rounding_size=0.03",
                                    fc="#f6f8fa", ec="#d0d7de", transform=ax.transAxes))
        y = 0.8
        for line in c["eq"]:
            small = line.startswith("W₀")
            ax.text(0.05, y, line, fontsize=11 if small else 13.5, color=MUTED if small else INK,
                    family="DejaVu Sans", transform=ax.transAxes, va="center")
            y -= 0.2
        ax.text(0.05, 0.14, c["learned"], fontsize=12, color=c["color"], fontweight="bold", transform=ax.transAxes)

        # paramètres et résultats à 500 neurones
        ax = fig.add_subplot(gs[2, 2 * j:2 * j + 2]); ax.axis("off")
        star = c["label"] == "réservoir accordé"
        ax.add_patch(FancyBboxPatch((0.01, 0.04), 0.98, 0.92, boxstyle="round,pad=0.01,rounding_size=0.03",
                                    fc="#eaf2fb" if star else "white", ec=c["color"], lw=2 if star else 1,
                                    transform=ax.transAxes))
        ax.text(0.05, 0.74, "paramètres récurrents appris", fontsize=10.5, color=MUTED, transform=ax.transAxes)
        ax.text(0.05, 0.4, c["params"], fontsize=20, fontweight="bold", color=c["color"], transform=ax.transAxes)
        ax.text(0.42, 0.74, "erreur à 500 neurones (NMSE)", fontsize=10.5, color=MUTED, transform=ax.transAxes)
        for k, (t, short) in enumerate((("Mackey-Glass (+15)", "Mackey-Glass"), ("NARMA10", "NARMA10"))):
            v = res[t][c["label"]]
            ratio = res[t]["RNN"] / v
            txt = f"{short} : {v:.1e}"
            if c["label"] == "réservoir accordé":
                txt += f"   ×{ratio:.0f} vs RNN"
            ax.text(0.42, 0.45 - 0.3 * k, txt, fontsize=12, color=INK, fontweight="bold" if star else "normal",
                    transform=ax.transAxes)
    fig.savefig(out_path("figures", "pitch.png"), dpi=110, facecolor="white")
    print("Figure : results/figures/pitch.png")


if __name__ == "__main__":
    main()
