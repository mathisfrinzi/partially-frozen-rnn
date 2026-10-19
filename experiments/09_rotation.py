"""
Parties 1.5 et 3 — Spectre contre directions.

    Rotation : W = Q · W₀ · Qᵀ (Q orthogonale apprise) -> spectre du réservoir conservé exactement
    Spectre  : W = U · diag(σ ⊙ g) · Vᵀ (g appris)          -> directions du réservoir conservées

    python experiments/09_rotation.py         (~20 min sur 2 cœurs ; instantané si en cache)

Plus une analyse : comment l'entraînement d'un RNN classique modifie-t-il le spectre du réservoir ?
Sorties : results/data/rotation.json, results/figures/rotation.png, results/figures/spectra.png
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn import core, params as P, tasks
from hfrnn.core import out_path
from hfrnn.runner import run_specs, grid, scores, infos, table

MG, NA, MEM = "Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)"
SEEDS = {MG: range(8), NA: range(8), MEM: range(5)}
FREE_SLOW = {"kind": "convex", "alpha": 1.0, "lr_mult": 0.1, "lr_mult_in": 1.0}
MODELS = [("réservoir classique", {"kind": "A"}, "gray"),
          ("RNN", {"kind": "convex", "alpha": 1.0}, "tab:red"),
          ("RNN freiné", FREE_SLOW, "tab:brown"),
          ("réservoir tourné (W_in tourné)", {"kind": "rotation", "win": "rotate"}, "tab:green"),
          ("réservoir tourné (W_in libre)", {"kind": "rotation", "win": "free"}, "tab:olive"),
          ("réservoir accordé", {"kind": "spectrum"}, "tab:blue")]


def spectra_analysis():
    """Spectre du réservoir, puis après entraînement d'un RNN classique et de la variante lente (seed 0)."""
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    out = {}
    th = np.linspace(0, 2 * np.pi, 200)
    for j, t in enumerate((MG, NA, MEM)):
        u, y = tasks.load(t)
        Win, W = core.make_reservoir(np.random.default_rng(0))
        mats = {"réservoir (départ)": W}
        for lab, prm in (("RNN entraîné", P.Convex(1.0)),
                         ("RNN freiné", P.Convex(1.0, lr_mult=0.1, lr_mult_in=1.0))):
            P.train(prm, u, y, Win, W, seed=0)
            mats[lab] = prm.compose(Win, W)[1]
        ax, ax2 = axes[0, j], axes[1, j]
        ax.plot(np.cos(th), np.sin(th), color="lightgray", lw=1)
        out[t] = {}
        for (lab, M), c, mk in zip(mats.items(), ("gray", "tab:red", "tab:brown"), ("o", "x", "+")):
            ev = np.linalg.eigvals(M)
            sv = np.linalg.svd(M, compute_uv=False)
            ax.scatter(ev.real, ev.imag, s=14, color=c, marker=mk, label=lab, alpha=0.8)
            ax2.plot(np.arange(1, len(sv) + 1), sv, color=c, label=lab)
            rel = float(np.linalg.norm(M - W) / np.linalg.norm(W))
            out[t][lab] = {"rayon_spectral": float(np.abs(ev).max()), "sv_max": float(sv[0]),
                           "changement_relatif_W": rel}
            print(f"  {t:20s} {lab:32s} rayon spectral {np.abs(ev).max():.3f}   σ_max {sv[0]:.3f}   "
                  f"‖W − W₀‖/‖W₀‖ = {rel:.2f}")
        ax.set_aspect("equal"); ax.set_title(f"{t} : valeurs propres"); ax.legend(fontsize=8)
        ax2.set_yscale("log"); ax2.set_xlabel("rang"); ax2.set_ylabel("valeur singulière")
        ax2.set_title(f"{t} : valeurs singulières"); ax2.legend(fontsize=8)
    fig.suptitle("Ce que l'entraînement change dans la matrice récurrente (seed 0)", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path("figures", "spectra.png"), dpi=110)
    return out


def main():
    specs = []
    for t, seeds in SEEDS.items():
        specs += grid([m for _, m, _ in MODELS], [t], list(seeds))
    R = run_specs(specs)
    out = {"comparaison": {}}
    for t, seeds in SEEDS.items():
        seeds = list(seeds)
        rows = [(l, scores(R, dict(m, task=t), seeds)) for l, m, _ in MODELS]
        out["comparaison"][t] = table(rows, "RNN", f"[rotation] {t}")
        for l, m, _ in MODELS:
            if m["kind"] == "rotation":
                d = infos(R, dict(m, task=t), seeds, "rot_dist")
                print(f"    {l:40s} écart de Q à l'identité (‖Q − I‖/√N) : {d.mean():.2f}")

    print("\nAnalyse des spectres :")
    out["spectres"] = spectra_analysis()
    with open(out_path("data", "rotation.json"), "w") as f:
        json.dump(out, f, indent=1)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    for ax, t in zip(axes, SEEDS):
        rows = out["comparaison"][t]
        ax.bar(range(len(rows)), [r["mean"] for r in rows], yerr=[r["se"] for r in rows], capsize=3,
               color=[c for _, _, c in MODELS])
        ax.axhline(rows[1]["mean"], color="tab:red", ls="--", lw=1)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([r["label"] for r in rows], rotation=35, ha="right", fontsize=8)
        ax.set_yscale("log"); ax.set_ylabel("erreur de test (NMSE, log)"); ax.set_title(f"{t} — {len(SEEDS[t])} seeds")
    fig.suptitle("Spectre contre directions. Ligne rouge : RNN.", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path("figures", "rotation.png"), dpi=120)
    print("\nFigures : results/figures/rotation.png, results/figures/spectra.png")


if __name__ == "__main__":
    main()
