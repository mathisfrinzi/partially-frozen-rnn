"""
Partie 8.3 : le réservoir accordé est-il vraiment 74 fois meilleur que le GRU sur l'amplificateur de puissance ?

Sur la carte des tâches (1 500 pas, un learning rate par modèle), le réservoir accordé est le seul modèle à réussir
la tâche « amplificateur de puissance » (NMSE 0.0035, contre 0.20 à 0.75 pour tous les autres). Un écart aussi grand
peut venir d'un réglage défavorable aux autres modèles. On donne donc au GRU et au RNN plus de chances : deux
learning rates et un entraînement 2,7 fois plus long (4 000 pas), et la même chose au réservoir accordé.

    python experiments/33_amplifier_control.py            # GPU (~20 min) ; marche aussi sur CPU (plus lent)
    python experiments/33_amplifier_control.py --quick    # vérification rapide
    python experiments/33_amplifier_control.py --report
Sorties : results/data/amplifier_control.json, results/figures/amplifier_control.png
"""
import argparse
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn import core, task_map

TASK = "Amplificateur de puissance"
LR = core.LR
CONFIGS = [
    ("réservoir accordé", {"kind": "spectrum"}, "tab:blue"),
    ("GRU, init soignée, lr 2e-3", {"kind": "gru", "rec_init": "ortho"}, "tab:orange"),
    ("GRU, init soignée, lr 5e-4", {"kind": "gru", "rec_init": "ortho", "lr": 5e-4}, "goldenrod"),
    ("RNN, lr global × 0.1", {"kind": "convex", "alpha": 1.0, "lr": 0.1 * LR}, "tab:red"),
    ("RNN, lr standard", {"kind": "convex", "alpha": 1.0}, "salmon"),
    ("réservoir accordé à portes", {"kind": "gated_tuned", "cell": "gru"}, "tab:cyan"),
]
STEPS = [1500, 4000]


def run(a, path, res):
    import torch
    from hfrnn import torch_models as TM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    u, y, nt = task_map.load(TASK)
    print(f"appareil : {dev} | N = {a.n} | {a.seeds} seeds | pas : {a.steps}", flush=True)
    t0 = time.time()
    for seed in range(a.seeds):
        Win, W = TM.make_reservoir(a.n, n_in=2, rho=core.RHO, in_scale=core.IN_SCALE / np.sqrt(2),
                                   sparsity=core.SPARSITY, seed=seed)
        for label, kw, _ in CONFIGS:
            cell = res["results"].setdefault(label, {})
            if str(seed) in cell:
                continue
            last = max(a.steps)
            s, info = TM.train_regression(u, y, Win, W, seed, steps=last, device=dev, n_train=nt,
                                          eval_at=tuple(a.steps), **dict(kw))
            cell[str(seed)] = {str(k): float(v) for k, v in info["checkpoints"].items()}
            print(f"[{(time.time() - t0) / 60:6.1f} min] seed {seed} {label:32s} "
                  + "  ".join(f"{k} pas : {v:.3g}" for k, v in cell[str(seed)].items()), flush=True)
            with open(path, "w") as f:
                json.dump(res, f, indent=1, ensure_ascii=False)


def report(res):
    R = res["results"]
    steps = [str(s) for s in res["args"]["steps"]]
    out = {}
    print(f"\nContrôle, {TASK}, N = {res['args']['n']} : NMSE de test (moyenne géométrique sur les seeds)")
    for label, _, _ in CONFIGS:
        if label not in R or not R[label]:
            continue
        row = {k: float(np.exp(np.mean([np.log(v[k]) for v in R[label].values()]))) for k in steps}
        out[label] = row
        print(f"  {label:32s} " + "   ".join(f"{k} pas : {row[k]:.3g}" for k in steps) + f"   ({len(R[label])} seeds)")
    with open(core.out_path("data", "amplifier_control_summary.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    w = 0.8 / len(steps)
    for j, k in enumerate(steps):
        ax.bar(np.arange(len(out)) + (j - 0.5) * w, [out[l][k] for l in out], w,
               color=[c for l, _, c in CONFIGS if l in out], alpha=0.55 + 0.45 * j, label=f"{k} pas")
    ax.set_xticks(range(len(out))); ax.set_xticklabels(list(out), rotation=20, ha="right", fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("NMSE de test (log)"); ax.legend(fontsize=8)
    ax.set_title(f"Contrôle de la partie 8.5 : {TASK} ({res['args']['n']} neurones)", fontsize=10)
    fig.tight_layout(); fig.savefig(core.out_path("figures", "amplifier_control.png"), dpi=110)
    print("\nFigure : results/figures/amplifier_control.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    a.steps = STEPS
    if a.quick:
        a.n, a.seeds, a.steps = 16, 1, [5, 10]
    path = core.out_path("data", f"amplifier_control{'_quick' if a.quick else ''}.json")
    res = {"args": {"n": a.n, "steps": a.steps}, "results": {}}
    if path.exists():
        old = json.load(open(path))
        if old["args"] == res["args"]:
            res = old
    if not a.report:
        run(a, path, res)
    report(res)


if __name__ == "__main__":
    main()
