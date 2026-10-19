"""
Partie 4.4 — Exploration : combiner les idées des parties 1 à 3.

    accordé (partie 3)  : on n'apprend que les amplitudes g des directions de W₀
    sculpté (partie 2)  : on n'apprend qu'un masque d'atténuation m ∈ [0, 1] sur chaque connexion
    freiné  (partie 1)  : une correction libre ΔW, apprise 10 fois plus lentement
    intégral            : les poids d'entrée ne sont pas appris non plus, seulement masqués

    W = m ⊙ (U · diag(σ ⊙ g) · Vᵀ) + ΔW       (chaque terme présent ou non)

    python experiments/13_exploration.py        (~35 min sur 2 cœurs ; instantané si en cache)

Mêmes tâches et seeds que la partie 2 : Mackey-Glass (+15) et NARMA10 (8 seeds), mémoire (délai 15, 5 seeds).
Réservoir dense (toutes les connexions) pour les combinaisons, comme pour le réservoir sculpté.
Sorties : results/data/exploration.json, results/figures/exploration.png
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
from hfrnn.runner import run_specs, grid, scores, paired, table

MG, NA, MEM = "Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)"
SEEDS = {MG: range(8), NA: range(8), MEM: range(5)}
D = {"sparsity": 1.0}
RNNS = ["RNN", "RNN, lr global × 0.3", "RNN, lr global × 0.1"]


def combo(**kw):
    return dict({"kind": "combo"}, **kw)


# (étiquette, spec, couleur, groupe)
REFS = [
    ("réservoir classique, dense", dict({"kind": "A"}, **D), "silver", "réf"),
    ("RNN", {"kind": "convex", "alpha": 1.0}, "tab:red", "réf"),
    ("RNN, lr global × 0.3", {"kind": "convex", "alpha": 1.0, "lr": 0.3 * LR}, "salmon", "réf"),
    ("RNN, lr global × 0.1", {"kind": "convex", "alpha": 1.0, "lr": 0.1 * LR}, "salmon", "réf"),
    ("P1 : RNN freiné", {"kind": "convex", "alpha": 1.0, "lr_mult": 0.1, "lr_mult_in": 1.0}, "tab:brown", "réf"),
    ("P2 : réservoir sculpté intégral, dense",
     dict({"kind": "supermask", "mode": "sigmoid", "win": "mask", "mult": 50.0}, **D), "tab:purple", "réf"),
    ("P3 : réservoir accordé", {"kind": "spectrum"}, "tab:blue", "réf"),
]
NEW = [
    ("accordé, dense", dict({"kind": "spectrum"}, **D), "tab:blue"),
    ("accordé intégral", combo(spectrum=True, win="mask"), "tab:cyan"),
    ("accordé intégral, dense", combo(spectrum=True, win="mask", **D), "tab:cyan"),
    ("accordé + sculpté, dense", combo(spectrum=True, mask=True, **D), "tab:green"),
    ("accordé + sculpté intégral, dense", combo(spectrum=True, mask=True, win="mask", **D), "tab:green"),
    ("accordé + freiné, dense", combo(spectrum=True, slow=0.1, **D), "tab:olive"),
    ("sculpté + freiné, dense", combo(mask=True, slow=0.1, **D), "tab:pink"),
    ("accordé + sculpté + freiné, dense", combo(spectrum=True, mask=True, slow=0.1, **D), "black"),
]
MODELS = REFS + [(l, s, c, "nouveau") for l, s, c in NEW]


def main():
    specs = []
    for t, seeds in SEEDS.items():
        specs += grid([m for _, m, _, _ in MODELS], [t], list(seeds))
    R = run_specs(specs)

    out = {}
    for t, seeds in SEEDS.items():
        seeds = list(seeds)
        sc = {l: scores(R, dict(m, task=t), seeds) for l, m, _, _ in MODELS}
        best_rnn = min(RNNS, key=lambda k: np.mean(np.log10(sc[k])))
        champs = [l for l, _, _, g in REFS if l.startswith("P")]
        best_champ = min(champs, key=lambda k: np.mean(np.log10(sc[k])))
        rows = [(l, sc[l]) for l, _, _, _ in MODELS]
        out[t] = {"table": table(rows, best_rnn, f"[exploration] {t} — référence : {best_rnn}"),
                  "meilleur RNN": best_rnn, "meilleur des parties 1-3": best_champ, "contre le meilleur des parties 1-3": {}}
        print(f"    écart apparié au meilleur modèle des parties 1 à 3 ({best_champ}) :")
        for l, _, _ in NEW:
            d, se, _ = paired(sc[l], sc[best_champ])
            out[t]["contre le meilleur des parties 1-3"][l] = {"écart": d, "se": se}
            print(f"      {l:40s} {d:+.2f} ± {se:.2f}")
    with open(out_path("data", "exploration.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.6))
    for ax, t in zip(axes, SEEDS):
        rows = {r["label"]: r for r in out[t]["table"]}
        labs = [l for l, _, _, _ in MODELS]
        m = [rows[l]["mean"] for l in labs]
        ax.bar(range(len(labs)), m, yerr=[rows[l]["se"] for l in labs], capsize=3,
               color=[c for _, _, c, _ in MODELS], edgecolor=["none"] * len(REFS) + ["black"] * len(NEW))
        best = rows[out[t]["meilleur RNN"]]["mean"]
        ax.axhline(best, color="tab:red", ls="--", lw=1)
        ax.axvline(len(REFS) - 0.5, color="black", lw=0.8)
        for i, l in enumerate(labs[len(REFS):], start=len(REFS)):
            ax.text(i, (m[i] + rows[l]["se"]) * 1.15, f"×{best / m[i]:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=55, ha="right", fontsize=7.5)
        ax.set_yscale("log"); ax.set_ylabel("erreur de test (NMSE, log)")
        ax.set_title(f"{t} — {len(SEEDS[t])} seeds", fontsize=11)
    fig.suptitle("Partie 4 — combinaisons (à droite du trait) contre les modèles des parties 1 à 3. "
                 "Tirets rouges : meilleur RNN ; « ×k » : k fois moins d'erreur que lui.", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path("figures", "exploration.png"), dpi=120)
    print("\nFigure : results/figures/exploration.png")


if __name__ == "__main__":
    main()
