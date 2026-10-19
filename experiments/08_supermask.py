"""
Partie 2 — Supermasque : les poids aléatoires ne changent jamais, on apprend seulement un masque.

    W = m ⊙ W₀        m continu dans [0, 1] (sigmoïde) ou binaire (top-k)

    python experiments/08_supermask.py        (~25 min sur 2 cœurs ; instantané si en cache)

Tâches : Mackey-Glass (+15) et NARMA10 (8 seeds), mémoire (délai 15, 5 seeds).
Réservoir « creux » : 10 % de connexions (comme en partie 1) ; « dense » : toutes les connexions.
Sorties : results/data/supermask.json, results/figures/supermask.png
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn.core import LR, out_path
from hfrnn.runner import run_specs, grid, scores, infos, table

MG, NA, MEM = "Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)"
SEEDS = {MG: range(8), NA: range(8), MEM: range(5)}
D = {"sparsity": 1.0}                           # réservoir dense
FREE_SLOW = {"kind": "convex", "alpha": 1.0, "lr_mult": 0.1, "lr_mult_in": 1.0}

REFS = [("réservoir classique", {"kind": "A"}, "gray"),
        ("réservoir classique, dense", dict({"kind": "A"}, **D), "silver"),
        ("RNN", {"kind": "convex", "alpha": 1.0}, "tab:red"),
        ("RNN, lr global × 0.3", {"kind": "convex", "alpha": 1.0, "lr": 0.3 * LR}, "salmon"),
        ("RNN, départ dense", dict({"kind": "convex", "alpha": 1.0}, **D), "salmon"),
        ("RNN freiné", FREE_SLOW, "tab:brown"),
        ("RNN freiné, départ dense", dict(FREE_SLOW, **D), "tab:brown")]
MASKS = [("réservoir sculpté, creux", {"kind": "supermask", "mode": "sigmoid", "win": "free", "mult": 50.0}),
         ("réservoir sculpté, dense", dict({"kind": "supermask", "mode": "sigmoid", "win": "free", "mult": 50.0}, **D)),
         ("réservoir sculpté, dense (lr des scores ÷ 5)",
          dict({"kind": "supermask", "mode": "sigmoid", "win": "free", "mult": 10.0}, **D)),
         ("réservoir sculpté intégral, dense",
          dict({"kind": "supermask", "mode": "sigmoid", "win": "mask", "mult": 50.0}, **D)),
         ("contrôle : réservoir repondéré, dense",
          dict({"kind": "supermask", "mode": "gain", "win": "mask", "mult": 2.5}, **D)),
         ("réservoir élagué 50 %, dense",
          dict({"kind": "supermask", "mode": "topk", "keep": 0.5, "win": "free", "mult": 10.0, "rescale": True}, **D))]


def main():
    models = REFS + [(l, m, "tab:pink" if l.startswith("contrôle") else "tab:purple") for l, m in MASKS]
    specs = []
    for t, seeds in SEEDS.items():
        specs += grid([m for _, m, _ in models], [t], list(seeds))
    R = run_specs(specs)

    out = {}
    for t, seeds in SEEDS.items():
        seeds = list(seeds)
        rows = [(l, scores(R, dict(m, task=t), seeds)) for l, m, _ in models]
        out[t] = {"table": table(rows, "RNN", f"[supermasque] {t}"), "masks": {}}
        print("    masques : fraction gardée (m > 0.5), masque moyen, NMSE si on binarise le masque après coup")
        for l, m in MASKS:
            sp = dict(m, task=t)
            kept, mean = infos(R, sp, seeds, "kept"), infos(R, sp, seeds, "mask_mean")
            binz = infos(R, sp, seeds, "nmse_binarise")
            full = scores(R, sp, seeds)
            msg = f"      {l:46s} gardé {kept.mean():.2f}  moyen {mean.mean():.2f}"
            if not np.all(np.isnan(binz)):
                msg += f"  NMSE binarisé {binz.mean():.4g} (contre {full.mean():.4g})"
            print(msg)
            out[t]["masks"][l] = {"kept": kept.tolist(), "mask_mean": mean.tolist(),
                                  "nmse_binarise": [None if np.isnan(x) else float(x) for x in binz]}
    with open(out_path("data", "supermask.json"), "w") as f:
        json.dump(out, f, indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))
    for ax, t in zip(axes, SEEDS):
        rows = out[t]["table"]
        ax.bar(range(len(rows)), [r["mean"] for r in rows], yerr=[r["se"] for r in rows], capsize=3,
               color=[c for _, _, c in models])
        ax.axhline(rows[2]["mean"], color="tab:red", ls="--", lw=1)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r["label"] for r in rows], rotation=50, ha="right", fontsize=7.5)
        ax.set_yscale("log"); ax.set_ylabel("erreur de test (NMSE, log)"); ax.set_title(f"{t} — {len(SEEDS[t])} seeds")
    fig.suptitle("Supermasque (violet) : seules les entrées et un masque sur les poids aléatoires sont appris. "
                 "Ligne rouge : RNN.", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path("figures", "supermask.png"), dpi=120)
    print("\nFigure : results/figures/supermask.png")


if __name__ == "__main__":
    main()
