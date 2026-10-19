"""
Robustesse : le résultat principal tient-il quand on change la taille ou le rayon spectral ?

    python experiments/04_robustness.py          (~1 h)

Réservoir classique, RNN ralenti × 0.1, RNN freiné, réservoir retouché et mélange réglé, pour N ∈ {50, 100, 200} et ρ ∈ {0.6, 0.9, 1.2}.
5 seeds, Mackey-Glass et NARMA10.
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
from hfrnn.runner import run_specs, grid, scores, paired

MG, NA = "Mackey-Glass (+15)", "NARMA10"
SEEDS = list(range(5))
VARIANTS = [("N = 50", {"n": 50}), ("N = 100", {}), ("N = 200", {"n": 200}),
            ("ρ = 0.6", {"rho": 0.6}), ("ρ = 1.2", {"rho": 1.2})]
MODELS = [("réservoir classique", {"kind": "A"}, "gray"),
          ("RNN", {"kind": "convex", "alpha": 1.0}, "tab:red"),
          ("RNN ralenti × 0.1", {"kind": "convex", "alpha": 1.0, "lr_mult": 0.1}, "tab:orange"),
          ("RNN freiné", {"kind": "convex", "alpha": 1.0, "lr_mult": 0.1, "lr_mult_in": 1.0},
           "tab:brown"),
          ("réservoir retouché", {"kind": "sparse"}, "tab:purple"),
          ("mélange réglé (α appris)", {"kind": "lalpha", "alpha0": 0.1}, "tab:cyan")]


def main():
    R = run_specs(grid([dict(m, **v) for _, m, _ in MODELS for _, v in VARIANTS], [MG, NA], SEEDS))
    out = {}
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for ax, t in zip(axes, (MG, NA)):
        out[t] = {}
        print(f"\n{t} (5 seeds)")
        print(f"{'':10s}" + "".join(f"{ml[:19]:>20s}" for ml, _, _ in MODELS))
        for vl, v in VARIANTS:
            sc = {ml: scores(R, dict(m, task=t, **v), SEEDS) for ml, m, _ in MODELS}
            cells, gaps = [], {}
            for ml, _, _ in MODELS:
                d, se, _ = paired(sc[ml], sc["RNN"])
                gaps[ml] = [float(d), float(se)]
                cells.append(f"{sc[ml].mean():9.6f} ({d:+.2f})")
            print(f"{vl:10s}" + "".join(f"{c:>20s}" for c in cells))
            out[t][vl] = {"nmse": {ml: sc[ml].tolist() for ml in sc}, "écart au RNN": gaps}
        print("  (entre parenthèses : écart apparié log10(NMSE / RNN))")
        w = 0.14
        for k, (ml, _, c) in enumerate(MODELS):
            m = [np.mean(out[t][vl]["nmse"][ml]) for vl, _ in VARIANTS]
            e = [np.std(out[t][vl]["nmse"][ml], ddof=1) / np.sqrt(len(SEEDS)) for vl, _ in VARIANTS]
            ax.bar(np.arange(len(VARIANTS)) + (k - 2.5) * w, m, yerr=e, width=w, capsize=2, color=c, label=ml)
        ax.set_xticks(range(len(VARIANTS))); ax.set_xticklabels([vl for vl, _ in VARIANTS])
        ax.set_yscale("log"); ax.set_ylabel("NMSE test"); ax.set_title(t); ax.legend(fontsize=8)
    fig.suptitle("Robustesse à la taille du réseau (N) et au rayon spectral du réservoir (ρ)", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path("figures", "robustness.png"), dpi=120)
    with open(out_path("data", "robustness.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nFigure : results/figures/robustness.png")


if __name__ == "__main__":
    main()
