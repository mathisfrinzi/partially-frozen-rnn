"""
Partie 8.3 — Données réelles : trois séries temporelles publiques, prédites à un horizon fixe.

    ETTh1                 température d'huile d'un transformateur, horaire, variation à 24 h (17 396 pas)
    Melbourne             température minimale quotidienne, variation à 1 jour (3 649 pas)
    Taches solaires       nombre mensuel de taches solaires, variation à 1 mois (2 819 pas)

Entrée : la valeur actuelle s(t). Cible : la variation s(t + h) − s(t). Même protocole que le reste de l'étude
(100 neurones, 1 500 pas d'entraînement, readout ridge), 75 % de la série pour l'entraînement, 25 % pour le test.
Référence naïve : la persistance « s(t + h) = s(t) », c'est-à-dire une variation prédite nulle.

    python experiments/14_real_data.py      (~40 min sur 2 cœurs ; instantané si en cache)
Sorties : results/data/real_data.json, results/figures/real_data.png
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn import core, tasks
from hfrnn.core import LR, out_path
from hfrnn.runner import run_specs, grid, scores, table

SEEDS = list(range(5))
D = {"sparsity": 1.0}
RNNS = ["RNN", "RNN, lr global × 0.3", "RNN, lr global × 0.1"]
MODELS = [
    ("réservoir classique", {"kind": "A"}, "gray"),
    ("RNN", {"kind": "convex", "alpha": 1.0}, "tab:red"),
    ("RNN, lr global × 0.3", {"kind": "convex", "alpha": 1.0, "lr": 0.3 * LR}, "salmon"),
    ("RNN, lr global × 0.1", {"kind": "convex", "alpha": 1.0, "lr": 0.1 * LR}, "salmon"),
    ("RNN freiné", {"kind": "convex", "alpha": 1.0, "lr_mult": 0.1, "lr_mult_in": 1.0}, "tab:brown"),
    ("réservoir corrigé de rang 4", {"kind": "lowrank", "rank": 4}, "tab:olive"),
    ("réservoir sculpté intégral, dense",
     dict({"kind": "supermask", "mode": "sigmoid", "win": "mask", "mult": 50.0}, **D), "tab:purple"),
    ("réservoir accordé", {"kind": "spectrum"}, "tab:blue"),
]


def persistence(name):
    """NMSE de la prévision naïve (variation nulle) sur la partie test, dans la normalisation de la cible."""
    u, y = tasks.load(name)
    _, raw_y = tasks.TASKS[name]()
    nt = core.n_train(u)
    pred = (0.0 - raw_y[:nt].mean()) / raw_y[:nt].std()
    return float(np.mean((pred - y[nt:]) ** 2) / np.var(y[nt:]))


def main():
    specs = grid([m for _, m, _ in MODELS], tasks.REAL, SEEDS)
    R = run_specs(specs)
    out = {}
    for t in tasks.REAL:
        sc = {l: scores(R, dict(m, task=t), SEEDS) for l, m, _ in MODELS}
        best = min(RNNS, key=lambda k: np.mean(np.log10(sc[k])))
        out[t] = {"table": table([(l, sc[l]) for l, _, _ in MODELS], best, f"[données réelles] {t}"),
                  "meilleur RNN": best, "persistance": persistence(t)}
        print(f"    prévision naïve (variation nulle) : NMSE = {out[t]['persistance']:.4g}")
    with open(out_path("data", "real_data.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    for ax, t in zip(axes, tasks.REAL):
        rows = {r["label"]: r for r in out[t]["table"]}
        labs = [l for l, _, _ in MODELS]
        m = [rows[l]["mean"] for l in labs]
        ax.bar(range(len(labs)), m, yerr=[rows[l]["se"] for l in labs], capsize=3, color=[c for _, _, c in MODELS])
        ax.axhline(rows[out[t]["meilleur RNN"]]["mean"], color="tab:red", ls="--", lw=1, label="meilleur RNN")
        ax.axhline(out[t]["persistance"], color="black", ls=":", lw=1.2, label="prévision naïve (variation nulle)")
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=40, ha="right", fontsize=8)
        ax.set_yscale("log"); ax.set_ylabel("erreur de test (NMSE, log)"); ax.set_title(t, fontsize=11)
        ax.legend(fontsize=8)
    fig.suptitle(f"Partie 8 — séries réelles, 100 neurones, {len(SEEDS)} seeds. Plus bas = mieux.", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path("figures", "real_data.png"), dpi=120)
    print("\nFigure : results/figures/real_data.png")


if __name__ == "__main__":
    main()
