"""
Partie 4.7 — Passage à l'échelle : le réservoir accordé n'apprend que N amplitudes (plus W_in), alors qu'un RNN en
apprend N². Que se passe-t-il quand N grandit (100 → 2 000) ?

Même protocole que le reste de l'étude (1 500 pas, readout ridge), version PyTorch, 3 seeds, 3 tâches.
Le RNN a deux learning rates (× 0.1 et × 0.03, les meilleurs à grande taille) ; on garde le meilleur.

    python experiments/16_scaling.py                     # GPU conseillé (~2 h 30 sur un T4)
    python experiments/16_scaling.py --sizes 100 200     # sous-ensemble
    python experiments/16_scaling.py --report

Résultats sauvegardés après chaque entraînement ; une relance reprend là où le calcul s'était arrêté.
"""
import argparse
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn import core, tasks

LR = core.LR
TASKS = ["Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)"]
MODELS = [
    ("réservoir classique", {"kind": "reservoir"}, "gray"),
    ("RNN, lr × 0.1", {"kind": "convex", "alpha": 1.0, "lr": 0.1 * LR}, "tab:red"),
    ("RNN, lr × 0.03", {"kind": "convex", "alpha": 1.0, "lr": 0.03 * LR}, "tab:red"),
    ("réservoir corrigé de rang 4", {"kind": "lowrank", "rank": 4}, "tab:olive"),
    ("réservoir sculpté intégral, dense",
     {"kind": "supermask", "mode": "sigmoid", "win": "mask", "mult": 50.0, "sparsity": 1.0}, "tab:purple"),
    ("réservoir accordé", {"kind": "spectrum"}, "tab:blue"),
]
FAMILY = {"RNN, lr × 0.1": "RNN (meilleur lr)", "RNN, lr × 0.03": "RNN (meilleur lr)"}
COLORS = {"réservoir classique": "gray", "RNN (meilleur lr)": "tab:red", "réservoir corrigé de rang 4": "tab:olive",
          "réservoir sculpté intégral, dense": "tab:purple", "réservoir accordé": "tab:blue"}


def run(a, path, res):
    import torch
    from hfrnn import torch_models as TM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"appareil : {dev} | N = {a.sizes} | {a.seeds} seeds | {a.steps} pas", flush=True)
    data = {t: tasks.load(t) for t in TASKS}
    todo = [(s, n, t, l) for s in range(a.seeds) for n in a.sizes for t in TASKS for l, _, _ in MODELS]
    t0, done = time.time(), 0
    for seed, n, task, label in todo:
        cell = res["results"].setdefault(str(n), {}).setdefault(task, {}).setdefault(label, {})
        if str(seed) in cell:
            continue
        kw = dict(next(m for l, m, _ in MODELS if l == label))
        sparsity = kw.pop("sparsity", core.SPARSITY)
        Win, W = TM.make_reservoir(n, rho=core.RHO, in_scale=core.IN_SCALE, sparsity=sparsity, seed=seed)
        u, y = data[task]
        score, info = TM.train_regression(u, y, Win, W, seed, steps=a.steps, device=dev, **kw)
        cell[str(seed)] = {"nmse": score, "n_params": info["n_params"]}
        done += 1
        left = sum(1 for s, nn, t, l in todo if str(s) not in res["results"].get(str(nn), {}).get(t, {}).get(l, {}))
        el = time.time() - t0
        print(f"[{el / 60:6.1f} min, reste ≈ {el / done * left / 60:5.0f} min] seed {seed} N = {n:5d} {task:20s} "
              f"{label:34s} NMSE = {score:.3g}", flush=True)
        with open(path, "w") as f:
            json.dump(res, f, indent=1)


def report(res):
    sizes = sorted(int(n) for n in res["results"])
    fams = ["réservoir classique", "RNN (meilleur lr)", "réservoir corrigé de rang 4",
            "réservoir sculpté intégral, dense", "réservoir accordé"]
    out = {}
    fig, axes = plt.subplots(2, len(TASKS), figsize=(6.4 * len(TASKS), 9.5))
    for j, t in enumerate(TASKS):
        curves = {f: [] for f in fams}
        npar = {f: [] for f in fams}
        for n in sizes:
            cell = res["results"][str(n)].get(t, {})
            seeds = min((len(v) for v in cell.values()), default=0)
            if seeds == 0 or len(cell) < len(MODELS):
                for f in fams:
                    curves[f].append(np.nan); npar[f].append(np.nan)
                continue
            S = {l: np.array([cell[l][str(s)]["nmse"] for s in range(seeds)]) for l in cell}
            rnn = min(("RNN, lr × 0.1", "RNN, lr × 0.03"), key=lambda l: np.mean(np.log10(S[l])))
            for f in fams:
                l = rnn if f == "RNN (meilleur lr)" else f
                curves[f].append(float(np.exp(np.mean(np.log(S[l])))))      # moyenne géométrique
                npar[f].append(cell[l]["0"]["n_params"])
        out[t] = {"N": sizes, "NMSE": curves, "paramètres": npar}
        print(f"\n{t} : NMSE (moyenne géométrique sur les seeds)")
        print("  " + " " * 34 + "".join(f"N = {n:<9d}" for n in sizes))
        for f in fams:
            print(f"  {f:34s}" + "".join(f"{v:<13.3g}" for v in curves[f]))
        ax = axes[0, j]
        for f in fams:
            ax.plot(sizes, curves[f], marker="o", color=COLORS[f], label=f, lw=2 if f == "réservoir accordé" else 1.3)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("N (neurones)"); ax.set_ylabel("NMSE test")
        ax.set_title(t); ax.grid(alpha=0.3, which="both")
        ax = axes[1, j]
        for f in fams[1:]:
            ax.plot(sizes, npar[f], marker="o", color=COLORS[f], label=f)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("N (neurones)")
        ax.set_ylabel("paramètres récurrents appris"); ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Passage à l'échelle : erreur (haut) et paramètres appris (bas) en fonction de la taille du réseau",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(core.out_path("figures", "scaling.png"), dpi=110)
    with open(core.out_path("data", "scaling_summary.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("\nFigure : results/figures/scaling.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=[100, 200, 500, 1000, 2000])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=core.STEPS)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    path = core.out_path("data", "scaling.json")
    res = {"args": {"steps": a.steps}, "results": {}}
    if path.exists():
        old = json.load(open(path))
        if old["args"] == res["args"]:
            res = old
    if not a.report:
        run(a, path, res)
    report(res)


if __name__ == "__main__":
    main()
