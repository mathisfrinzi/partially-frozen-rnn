"""
GPU : le résultat principal tient-il avec un réseau 5 fois plus grand ?

Mackey-Glass (+15) et NARMA10, même protocole que la version numpy, N = 500 neurones par défaut.

    python experiments/05_gpu_regression.py                  # N = 500, 3 seeds
    python experiments/05_gpu_regression.py --n 1000
    python experiments/05_gpu_regression.py --quick          # vérification rapide (2 min)

Les résultats sont sauvegardés après chaque modèle ; une relance ne calcule que les modèles manquants.
"""
import argparse
import json
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn import core, tasks
from hfrnn import torch_models as TM

MODELS = [
    ("réservoir classique", {"kind": "reservoir"}),
    ("RNN", {"kind": "convex", "alpha": 1.0}),
    ("RNN, lr global × 0.1", {"kind": "convex", "alpha": 1.0, "lr": 0.1 * core.LR}),
    ("RNN, lr global × 0.03", {"kind": "convex", "alpha": 1.0, "lr": 0.03 * core.LR}),
    ("RNN, lr global × 0.01", {"kind": "convex", "alpha": 1.0, "lr": 0.01 * core.LR}),
    ("RNN freiné", {"kind": "convex", "alpha": 1.0, "lr_mult": 0.1, "lr_mult_in": 1.0}),
    ("mélange fixe α = 0.1", {"kind": "convex", "alpha": 0.1}),
    ("RNN ralenti × 0.1", {"kind": "convex", "alpha": 1.0, "lr_mult": 0.1}),
    ("réservoir corrigé de rang 4", {"kind": "lowrank", "rank": 4}),
    ("réservoir corrigé de rang 16", {"kind": "lowrank", "rank": 16}),
    ("réservoir retouché", {"kind": "sparse"}),
    ("mélange réglé (α appris)", {"kind": "lalpha", "alpha0": 0.1}),
    # partie 2 : supermasque (les poids aléatoires ne changent jamais)
    ("réservoir classique, dense", {"kind": "reservoir", "sparsity": 1.0}),
    ("RNN, départ dense", {"kind": "convex", "alpha": 1.0, "sparsity": 1.0}),
    ("réservoir sculpté, dense", {"kind": "supermask", "mode": "sigmoid", "win": "free", "mult": 50.0, "sparsity": 1.0}),
    ("réservoir sculpté intégral, dense",
     {"kind": "supermask", "mode": "sigmoid", "win": "mask", "mult": 50.0, "sparsity": 1.0}),
    # partie 3 : spectre contre directions
    ("réservoir tourné", {"kind": "rotation", "win": "rotate"}),
    ("réservoir accordé", {"kind": "spectrum"}),
    # partie 4 : combinaisons (W = m ⊙ [U diag(σ ⊙ g) Vᵀ ou W₀] + ΔW lent)
    ("réservoir accordé, dense", {"kind": "spectrum", "sparsity": 1.0}),
    ("accordé intégral", {"kind": "combo", "spectrum": True, "win": "mask"}),
    ("accordé + sculpté, dense", {"kind": "combo", "spectrum": True, "mask": True, "sparsity": 1.0}),
    ("accordé + sculpté intégral, dense",
     {"kind": "combo", "spectrum": True, "mask": True, "win": "mask", "sparsity": 1.0}),
    ("accordé + freiné, dense", {"kind": "combo", "spectrum": True, "slow": 0.1, "sparsity": 1.0}),
    ("sculpté + freiné, dense", {"kind": "combo", "mask": True, "slow": 0.1, "sparsity": 1.0}),
    ("accordé + sculpté + freiné, dense",
     {"kind": "combo", "spectrum": True, "mask": True, "slow": 0.1, "sparsity": 1.0}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", nargs="*", default=None,
                    help="ne lancer que les modèles dont le nom contient l'un de ces mots (ex. --only entrées 'lr global')")
    ap.add_argument("--tasks", nargs="*", default=["Mackey-Glass (+15)", "NARMA10"],
                    help="ex. --tasks 'Mackey-Glass (+15)' NARMA10 'Mémoire (délai 15)'")
    a = ap.parse_args()
    if a.quick:
        a.n, a.seeds, a.steps = 200, 2, 150
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"appareil : {dev} | N = {a.n} | {a.seeds} seeds | {a.steps} pas", flush=True)

    tag = f"n{a.n}" + ("_quick" if a.quick else "")
    path = core.out_path("data", f"gpu_regression_{tag}.json")
    # on complète les résultats déjà calculés (mêmes N, seeds et pas) au lieu de tout relancer
    res = {}
    if path.exists():
        old = json.load(open(path))
        if all(old["args"].get(k) == getattr(a, k) for k in ("n", "seeds", "steps", "rho")):
            res = old["results"]
            print("résultats existants repris :", {t: len(v) for t, v in res.items()}, flush=True)
    todo = [(l, kw) for l, kw in MODELS if a.only is None or any(w in l for w in a.only)]
    t0 = time.time()
    for task in a.tasks:
        u, y = tasks.load(task)
        res.setdefault(task, {})
        for label, kw in todo:
            if label in res[task] and a.only is None:
                continue                                   # déjà calculé
            sc = []
            for seed in range(a.seeds):
                kw2 = dict(kw)
                sparsity = kw2.pop("sparsity", core.SPARSITY)
                Win, W = TM.make_reservoir(a.n, rho=a.rho, in_scale=core.IN_SCALE, sparsity=sparsity, seed=seed)
                s, info = TM.train_regression(u, y, Win, W, seed, steps=a.steps, device=dev, **kw2)
                sc.append(s)
            res[task][label] = sc
            print(f"[{time.time() - t0:6.0f}s] {task:20s} {label:32s} NMSE = {np.mean(sc):.3g} "
                  f"± {np.std(sc, ddof=1) / np.sqrt(len(sc)) if len(sc) > 1 else 0:.2g}", flush=True)
            with open(path, "w") as f:
                json.dump({"args": vars(a), "results": res}, f, indent=1)

    report(res, a.n, a.seeds, tag)


def report(res, n, seeds, tag):
    """Tableau apparié contre le RNN et figure."""
    labels = [l for l, _ in MODELS if all(l in res[t] for t in res)]
    for task in res:
        ref = np.array(res[task]["RNN"])
        print(f"\n{task} (N = {n}) : écart apparié log10(NMSE / RNN)")
        for label in labels:
            v = np.array(res[task][label])
            d = np.log10(v / ref)
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            print(f"  {label:32s} {v.mean():.3g}   écart {d.mean():+.3f} ± {se:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(17, 5.5))
    for ax, task in zip(axes, res):
        m = [np.mean(res[task][l]) for l in labels]
        se = [np.std(res[task][l], ddof=1) / np.sqrt(seeds) if seeds > 1 else 0 for l in labels]
        ax.bar(range(len(labels)), m, yerr=se, capsize=4, color="tab:blue")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_yscale("log"); ax.set_ylabel("NMSE test"); ax.set_title(f"{task}, N = {n}")
    fig.tight_layout()
    fig.savefig(core.out_path("figures", f"gpu_regression_{tag}.png"), dpi=110)
    print(f"\nFigure : results/figures/gpu_regression_{tag}.png")


if __name__ == "__main__":
    main()
