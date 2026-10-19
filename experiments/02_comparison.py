"""
Comparaison principale : réservoir classique, RNN, et les façons de rester près du réservoir.

    python experiments/02_comparison.py          (~45 min sur 2 cœurs ; instantané si en cache)

Tâches : Mackey-Glass (+15) et NARMA10 (8 seeds), mémoire pure délai 15 (5 seeds).
Sorties : results/data/comparison.json et les figures results/figures/{headline, comparison, alpha_vs_lr}.png
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn.core import LR, out_path
from hfrnn.runner import run_specs, grid, scores, table

MG, NA, MEM = "Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)"
SEEDS = {MG: range(8), NA: range(8), MEM: range(5)}
FS = [0.03, 0.1, 0.3]

MODELS = [("réservoir classique", {"kind": "A"}),
          ("RNN", {"kind": "convex", "alpha": 1.0}),
          ("RNN, lr global × 0.3", {"kind": "convex", "alpha": 1.0, "lr": 0.3 * LR}),
          ("RNN, lr global × 0.1", {"kind": "convex", "alpha": 1.0, "lr": 0.1 * LR})]
MODELS += [(f"mélange fixe α = {f}", {"kind": "convex", "alpha": f}) for f in FS]
MODELS += [(f"RNN ralenti × {f}", {"kind": "convex", "alpha": 1.0, "lr_mult": f}) for f in FS]
MODELS += [("RNN freiné", {"kind": "convex", "alpha": 1.0, "lr_mult": 0.1, "lr_mult_in": 1.0})]
MODELS += [(f"réservoir corrigé de rang {r}", {"kind": "lowrank", "rank": r}) for r in (0, 4, 16)]
MODELS += [("réservoir retouché", {"kind": "sparse"})]
MODELS += [("mélange réglé (α appris)", {"kind": "lalpha", "alpha0": 0.1})]

COLORS = {"réservoir classique": "gray", "RNN": "tab:red", "RNN, lr global × 0.3": "salmon",
          "RNN, lr global × 0.1": "salmon", "réservoir retouché": "tab:purple", "mélange réglé (α appris)": "tab:cyan",
          "RNN freiné": "tab:brown"}


def color(label):
    if label in COLORS:
        return COLORS[label]
    return {"m": "tab:blue", "R": "tab:orange", "r": "tab:green"}[label[0]]


def main():
    specs = []
    for t, seeds in SEEDS.items():
        specs += grid([m for _, m in MODELS], [t], list(seeds))
    R = run_specs(specs)

    out = {}
    for t, seeds in SEEDS.items():
        rows = [(l, scores(R, dict(m, task=t), list(seeds))) for l, m in MODELS]
        out[t] = table(rows, "RNN", t)
    with open(out_path("data", "comparison.json"), "w") as f:
        json.dump(out, f, indent=1)

    # figure 1 : toutes les variantes
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    for ax, t in zip(axes, SEEDS):
        rows = out[t]
        labels = [r["label"] for r in rows]
        ax.bar(range(len(rows)), [r["mean"] for r in rows], yerr=[r["se"] for r in rows], capsize=3,
               color=[color(l) for l in labels])
        ax.axhline(rows[1]["mean"], color="tab:red", ls="--", lw=1)
        ax.set_xticks(range(len(rows))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yscale("log"); ax.set_ylabel("erreur de test (NMSE, échelle log)")
        ax.set_title(f"{t} — {len(SEEDS[t])} seeds")
    fig.suptitle("Plus bas = mieux. Ligne rouge : RNN classique.", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path("figures", "comparison.png"), dpi=120)

    # figure 2 : mélange fixe contre RNN ralenti
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, t in zip(axes, SEEDS):
        by = {r["label"]: r for r in out[t]}
        xs = FS + [1.0]
        for key, lab, c, ls in (("mélange fixe α = {}", "mélange fixe α = f", "tab:blue", "-"),
                                ("RNN ralenti × {}", "RNN ralenti × f (entrées et récurrence)", "tab:orange", "--")):
            rr = [by[key.format(f)] for f in FS] + [by["RNN"]]
            ax.errorbar(xs, [r["mean"] for r in rr], yerr=[r["se"] for r in rr], marker="o", capsize=3,
                        color=c, ls=ls, lw=2, label=lab)
        ax.axhline(by["réservoir classique"]["mean"], color="gray", ls=":", label="réservoir classique")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("f (f = 1 : RNN classique)")
        ax.set_ylabel("NMSE test"); ax.set_title(t); ax.grid(alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.suptitle("Mélange fixe α et RNN ralenti × α donnent les mêmes résultats", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path("figures", "alpha_vs_lr.png"), dpi=120)
    # figure 3 : synthèse (meilleur RNN réglé contre les trois façons de rester près du réservoir)
    keep = [("réservoir classique", "gray"), (None, "tab:red"), ("RNN freiné", "tab:brown"),
            ("mélange réglé (α appris)", "tab:cyan"), ("réservoir retouché", "tab:purple")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    for ax, t in zip(axes, SEEDS):
        by = {r["label"]: r for r in out[t]}
        best = min(("RNN", "RNN, lr global × 0.3", "RNN, lr global × 0.1"), key=lambda k: by[k]["mean"])
        labs = [l if l else best for l, _ in keep]
        lr_txt = "lr standard" if best == "RNN" else best.replace("RNN, lr global", "lr")
        names = ["réservoir\nclassique", f"meilleur RNN\n({lr_txt})", "RNN freiné",
                 "mélange réglé\n(α appris)", "réservoir\nretouché"]
        means = [by[l]["mean"] for l in labs]
        ax.bar(range(len(labs)), means, yerr=[by[l]["se"] for l in labs], capsize=4, color=[c for _, c in keep])
        for i, l in enumerate(labs[2:], start=2):
            ax.text(i, (by[l]["mean"] + by[l]["se"]) * 1.25, f"×{by[best]['mean'] / by[l]['mean']:.1f}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_ylim(min(means) / 3, max(means) * 3)
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(names, fontsize=9)
        ax.set_yscale("log"); ax.set_ylabel("erreur de test (NMSE, log)"); ax.set_title(t, fontsize=12)
    fig.suptitle("Plus bas = mieux. « ×k » : k fois moins d'erreur que le meilleur RNN.", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path("figures", "headline.png"), dpi=120)
    print("\nFigures : results/figures/{headline, comparison, alpha_vs_lr}.png")


if __name__ == "__main__":
    main()
