"""
Partie 4.7 — Réservoir accordé rapide : de très grands réseaux.

Le réservoir accordé n'apprend que N amplitudes, mais chaque pas coûte N² (multiplication par une matrice dense).
Avec des directions de Fourier, W = P₁ · F⁻¹ · diag(λ ⊙ g) · F · P₂ s'applique en N·log N (FFT) : on peut monter
à des dizaines de milliers de neurones. Le readout ridge passe en forme duale quand N dépasse le nombre de pas
d'entraînement.

Modèles : accordé rapide et le même réservoir sans entraînement, à N = 500, 2 000, 8 000 et 16 000 ;
références denses (réservoir accordé, RNN à deux learning rates) à N = 500 et 2 000.

    python experiments/19_fast_scaling.py                     # GPU (~1 h 30 sur un T4)
    python experiments/19_fast_scaling.py --sizes 500 2000    # sous-ensemble
    python experiments/19_fast_scaling.py --report
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
DENSE_MAX = 2000
MODELS = [
    ("réservoir de Fourier, non entraîné", {"kind": "fast_reservoir"}, "silver", None),
    ("accordé rapide (Fourier)", {"kind": "fast_tuned"}, "tab:green", None),
    ("réservoir accordé (dense)", {"kind": "spectrum"}, "tab:blue", DENSE_MAX),
    ("RNN, lr × 0.1", {"kind": "convex", "alpha": 1.0, "lr": 0.1 * LR}, "tab:red", DENSE_MAX),
    ("RNN, lr × 0.03", {"kind": "convex", "alpha": 1.0, "lr": 0.03 * LR}, "tab:red", DENSE_MAX),
]
FAMS = [("réservoir de Fourier, non entraîné", "silver"), ("accordé rapide (Fourier)", "tab:green"),
        ("réservoir accordé (dense)", "tab:blue"), ("RNN (meilleur lr)", "tab:red")]


def run(a, path, res):
    import torch
    from hfrnn import torch_models as TM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"appareil : {dev} | N = {a.sizes} | {a.seeds} seeds | {a.steps} pas", flush=True)
    data = {t: tasks.load(t) for t in TASKS}
    todo = [(s, n, t, l) for s in range(a.seeds) for n in a.sizes for t in TASKS for l, _, _, nmax in MODELS
            if nmax is None or n <= nmax]
    t0, done = time.time(), 0
    for seed, n, task, label in todo:
        cell = res["results"].setdefault(str(n), {}).setdefault(task, {}).setdefault(label, {})
        if str(seed) in cell:
            continue
        kw = dict(next(m for l, m, _, _ in MODELS if l == label))
        if kw["kind"].startswith("fast"):
            # même tirage de W_in que make_reservoir, sans construire de matrice dense N × N
            Win = np.random.default_rng(seed).uniform(-core.IN_SCALE, core.IN_SCALE, (n, 1))
            W = np.zeros((1, 1))
        else:
            Win, W = TM.make_reservoir(n, rho=core.RHO, in_scale=core.IN_SCALE, sparsity=core.SPARSITY, seed=seed)
        u, y = data[task]
        t1 = time.time()
        score, info = TM.train_regression(u, y, Win, W, seed, steps=a.steps, device=dev, **kw)
        cell[str(seed)] = {"nmse": score, "n_params": info["n_params"], "seconds": time.time() - t1}
        done += 1
        left = sum(1 for s, nn, t, l in todo if str(s) not in res["results"].get(str(nn), {}).get(t, {}).get(l, {}))
        el = time.time() - t0
        print(f"[{el / 60:6.1f} min, reste ≈ {el / done * left / 60:5.0f} min] seed {seed} N = {n:6d} {task:20s} "
              f"{label:36s} NMSE = {score:.3g}  ({time.time() - t1:.0f} s)", flush=True)
        with open(path, "w") as f:
            json.dump(res, f, indent=1)


def report(res):
    sizes = sorted(int(n) for n in res["results"])
    out = {}
    fig, axes = plt.subplots(2, len(TASKS), figsize=(6.4 * len(TASKS), 9.5))
    for j, t in enumerate(TASKS):
        cur = {f: {} for f, _ in FAMS}
        for n in sizes:
            cell = res["results"][str(n)].get(t, {})
            seeds = min((len(v) for v in cell.values()), default=0)
            if seeds == 0:
                continue
            S = {l: np.array([cell[l][str(s)]["nmse"] for s in range(seeds)]) for l in cell}
            sec = {l: np.mean([cell[l][str(s)]["seconds"] for s in range(seeds)]) for l in cell}
            for f, _ in FAMS:
                if f == "RNN (meilleur lr)":
                    cands = [l for l in ("RNN, lr × 0.1", "RNN, lr × 0.03") if l in S]
                    if not cands:
                        continue
                    l = min(cands, key=lambda k: np.mean(np.log10(S[k])))
                else:
                    l = f
                if l in S:
                    cur[f][n] = {"nmse": float(np.exp(np.mean(np.log(S[l])))), "secondes": float(sec[l]),
                                 "paramètres": cell[l]["0"]["n_params"]}
        out[t] = cur
        print(f"\n{t} : NMSE (moyenne géométrique), paramètres appris, secondes par entraînement")
        for f, _ in FAMS:
            row = "  ".join(f"N={n}: {v['nmse']:.3g} ({v['paramètres']} p., {v['secondes']:.0f} s)"
                            for n, v in sorted(cur[f].items()))
            print(f"  {f:36s} {row}")
        ax = axes[0, j]
        for f, c in FAMS:
            ns = sorted(cur[f])
            if ns:
                ax.plot(ns, [cur[f][n]["nmse"] for n in ns], marker="o", color=c, label=f,
                        lw=2 if "rapide" in f else 1.3)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("N (neurones)"); ax.set_ylabel("NMSE test")
        ax.set_title(t); ax.grid(alpha=0.3, which="both")
        ax = axes[1, j]
        for f, c in FAMS[1:]:
            ns = sorted(cur[f])
            if ns:
                ax.plot(ns, [cur[f][n]["secondes"] for n in ns], marker="o", color=c, label=f)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("N (neurones)")
        ax.set_ylabel("temps d'entraînement (s)"); ax.grid(alpha=0.3, which="both")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Partie 7 — réservoir accordé rapide : erreur (haut) et temps de calcul (bas) selon la taille",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(core.out_path("figures", "fast_scaling.png"), dpi=110)
    with open(core.out_path("data", "fast_scaling_summary.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("\nFigure : results/figures/fast_scaling.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="*", default=[500, 2000, 8000, 16000])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=core.STEPS)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    path = core.out_path("data", "fast_scaling.json")
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
