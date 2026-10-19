"""
Partie 5.1 — figure de synthèse des décompositions : écart apparié au réservoir accordé, modèle par tâche.

Lit results/data/decompositions_n100_summary.json (produit par `36_decompositions.py --report`).
Bleu : mieux que l'accordé ; rouge : moins bien ; gris : égal. Chiffres en gras : écart au-delà du seuil de bruit.

    python experiments/36b_decompositions_figure.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from hfrnn import core

ROWS = [
    ("réservoir accordé (valeurs singulières)", "Accordé (référence)"),
    ("Iwasawa : amplitudes (A)", "Iwasawa A"),
    ("accordé, valeurs propres (Jordan-Chevalley, partie semi-simple)", "Jordan-Chevalley, semi-simple"),
    ("Iwasawa : cisaillement (N)", "Iwasawa N (cisaillement)"),
    ("Iwasawa : amplitudes et cisaillement (A et N)", "Iwasawa A et N"),
    ("Jordan-Chevalley + partie nilpotente apprise", "Jordan-Chevalley + nilpotente"),
    ("réservoir tourné (rotations)", "Réservoir tourné"),
    ("RNN", "RNN"),
]
SHORT = {"Mackey-Glass (+15)": "Mackey-Glass", "NARMA10": "NARMA10", "Mémoire (délai 15)": "Mémoire",
         "Lorenz (+10)": "Lorenz", "Hénon (+3)": "Hénon"}
CMAP = LinearSegmentedColormap.from_list("div", ["#2f6fd0", "#f0efec", "#d0453b"])


def main():
    s = json.load(open(core.out_path("data", "decompositions_n100_summary.json")))
    tasks = [t for t in SHORT if t in s]
    G = [[s[t][k]["vs accordé"] for t in tasks] for k, _ in ROWS]
    V = [[s[t][k]["verdict"] for t in tasks] for k, _ in ROWS]
    P = [s[tasks[0]][k]["paramètres"] for k, _ in ROWS]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.imshow(G, cmap=CMAP, norm=TwoSlopeNorm(0, -0.8, 0.8), aspect="auto")
    for i, row in enumerate(G):
        for j, g in enumerate(row):
            sig = V[i][j] in ("mieux", "moins bien")
            txt = "réf." if i == 0 else (f"×{10 ** -g:.1f}" if g < 0 else f"÷{10 ** g:.1f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9.5, color="#1f2328",
                    weight="bold" if sig else "normal", alpha=1.0 if (sig or i == 0) else 0.55)
    ax.set_xticks(range(len(tasks)), [SHORT[t] for t in tasks])
    ax.set_yticks(range(len(ROWS)), [f"{lab}  ({p:,} par.)".replace(",", " ") for (_, lab), p in zip(ROWS, P)])
    ax.xaxis.tick_top()
    ax.tick_params(which="both", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(tasks))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(ROWS))], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    fig.suptitle("Partie 2.8 — erreur par rapport au réservoir accordé (100 neurones, 3 graines)",
                 fontsize=11, x=0.02, ha="left")
    fig.text(0.02, 0.015, "×2 : deux fois moins d'erreur (bleu)   ÷2 : deux fois plus (rouge)   "
             "gras : au-delà du bruit ; pâle : égal", fontsize=9, color="#59636e")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    out = core.out_path("figures", "decompositions_n100.png")
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()
