"""
Et si α était appris par rétropropagation ?  α = sigmoid(θ), un α pour W_in, un pour W.

    python experiments/03_learned_alpha.py       (~15 min)

On part de α = 0.1 ou α = 0.5 et on regarde où α arrive et le score obtenu, comparé aux α fixes.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn.core import out_path
from hfrnn.runner import run_specs, grid, scores, infos, table

MG, NA, MEM = "Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)"
SEEDS = {MG: range(8), NA: range(8), MEM: range(5)}
STARTS = [0.1, 0.5]
REFS = [("RNN", {"kind": "convex", "alpha": 1.0})] + \
       [(f"mélange fixe α = {a}", {"kind": "convex", "alpha": a}) for a in (0.03, 0.1, 0.3)]


def main():
    models = [m for _, m in REFS] + [{"kind": "lalpha", "alpha0": a} for a in STARTS]
    specs = []
    for t, seeds in SEEDS.items():
        specs += grid(models, [t], list(seeds))
    R = run_specs(specs)
    out = {}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (t, seeds) in zip(axes, SEEDS.items()):
        seeds = list(seeds)
        rows = [(l, scores(R, dict(m, task=t), seeds)) for l, m in REFS]
        rows += [(f"mélange réglé, départ α = {a}", scores(R, {"kind": "lalpha", "alpha0": a, "task": t}, seeds))
                 for a in STARTS]
        out[t] = {"table": table(rows, "RNN", t), "alpha_final": {}}
        for a in STARTS:
            sp = {"kind": "lalpha", "alpha0": a, "task": t}
            ai, ar = infos(R, sp, seeds, "alpha_in"), infos(R, sp, seeds, "alpha_rec")
            print(f"    départ {a} : α_in final = {ai.mean():.2f} ± {ai.std():.2f}, "
                  f"α_rec final = {ar.mean():.2f} ± {ar.std():.2f}")
            out[t]["alpha_final"][str(a)] = {"alpha_in": ai.tolist(), "alpha_rec": ar.tolist()}
        tab = out[t]["table"]
        ax.bar(range(len(tab)), [r["mean"] for r in tab], yerr=[r["se"] for r in tab], capsize=3,
               color=["tab:red"] + ["tab:blue"] * 3 + ["tab:green"] * len(STARTS))
        for i, a in enumerate(STARTS):
            ai = np.mean(out[t]["alpha_final"][str(a)]["alpha_in"])
            ar = np.mean(out[t]["alpha_final"][str(a)]["alpha_rec"])
            r = tab[len(REFS) + i]
            ax.text(len(REFS) + i, (r["mean"] + r["se"]) * 1.08, f"α_in → {ai:.2f}\nα_rec → {ar:.2f}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(tab))); ax.set_xticklabels([r["label"] for r in tab], rotation=30, ha="right",
                                                           fontsize=8)
        ax.set_yscale("log"); ax.set_ylabel("NMSE test"); ax.set_title(t)
    fig.suptitle("Mélange réglé (α appris), valeurs finales au-dessus des barres : α_in monte selon la tâche, α_rec reste bas", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path("figures", "learned_alpha.png"), dpi=120)
    with open(out_path("data", "learned_alpha.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nFigure : results/figures/learned_alpha.png")


if __name__ == "__main__":
    main()
