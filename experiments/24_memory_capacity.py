"""
Partie 4.3 — Pourquoi ça marche : la capacité mémoire de chaque modèle, avant et après entraînement.

La capacité mémoire (Jaeger, 2002) mesure combien de pas en arrière un réseau « se souvient » de son entrée : on fait
passer un signal aléatoire u(t) (i.i.d. uniforme, moyenne 0, variance 1, comme l'entrée normalisée des tâches) et on
demande à une lecture linéaire de retrouver u(t − k). La capacité au délai k est C_k = 1 − NMSE de cette lecture
sur une partie de test (0 = rien retrouvé, 1 = parfait). Sa somme sur k, MC = Σ C_k, est au plus N.

On mesure aussi une capacité non linéaire de degré 2 (Dambre et al., 2012) : retrouver P₂(u(t − k)) (polynôme de
Legendre de degré 2) et les produits u(t − i)·u(t − j). Mémoire linéaire et calcul non linéaire se partagent les
N dimensions de l'état : c'est le compromis mémoire–non-linéarité.

Chaque modèle est mesuré après un vrai entraînement sur chacune des trois tâches (même protocole que les parties 1
à 3 : 100 neurones, 1 500 pas), et les réservoirs non entraînés servent de point de départ.

    python experiments/24_memory_capacity.py              # N = 100, 3 seeds, CPU (~25 min sur 2 cœurs)
    python experiments/24_memory_capacity.py --report
Sorties : results/data/memory_capacity.json, results/figures/memory_capacity.png
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn import core, tasks

LR = core.LR
TASKS = ["Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)"]
UNTRAINED = [("réservoir classique", {"kind": "reservoir"}, "gray"),
             ("réservoir classique, dense", {"kind": "reservoir", "sparsity": 1.0}, "silver")]
TRAINED = [
    ("RNN", {"kind": "convex", "alpha": 1.0}, "tab:red"),
    ("RNN, lr global × 0.1", {"kind": "convex", "alpha": 1.0, "lr": 0.1 * LR}, "salmon"),
    ("RNN freiné", {"kind": "convex", "alpha": 1.0, "lr_mult": 0.1, "lr_mult_in": 1.0}, "tab:brown"),
    ("réservoir corrigé de rang 4", {"kind": "lowrank", "rank": 4}, "tab:olive"),
    ("réservoir sculpté intégral, dense",
     {"kind": "supermask", "mode": "sigmoid", "win": "mask", "mult": 50.0, "sparsity": 1.0}, "tab:purple"),
    ("réservoir tourné", {"kind": "rotation", "win": "rotate"}, "tab:pink"),
    ("réservoir accordé", {"kind": "spectrum"}, "tab:blue"),
]
COLORS = {l: c for l, _, c in UNTRAINED + TRAINED}
K_LIN, K_SQ, K_CROSS = 100, 30, 15          # délais mesurés : linéaires, carrés, produits croisés (i < j ≤ K_CROSS)
WASH, T_FIT, T_TEST = 200, 4000, 4000


def probe_input(seed):
    rng = np.random.default_rng(10_000 + seed)
    return rng.uniform(-np.sqrt(3), np.sqrt(3), WASH + T_FIT + T_TEST + K_LIN)


def capacities(states, u):
    """C = max(0, 1 − NMSE test) pour chaque cible ; lecture ridge apprise sur T_FIT pas, testée sur T_TEST autres."""
    import torch
    from hfrnn.torch_models import ridge
    K = K_LIN
    X = states[K:]                                            # on commence quand tous les délais existent
    x = u / np.sqrt(3)                                        # dans [−1, 1] pour les polynômes de Legendre
    T = len(X)
    lag = lambda k: x[K - k:K - k + T]                        # noqa: E731  u(t − k), aligné sur X
    targets = {f"lin{k}": lag(k) for k in range(1, K_LIN + 1)}
    targets |= {f"sq{k}": 1.5 * lag(k) ** 2 - 0.5 for k in range(0, K_SQ + 1)}
    targets |= {f"x{i}_{j}": lag(i) * lag(j) for i in range(0, K_CROSS + 1) for j in range(i + 1, K_CROSS + 1)}
    a, b = WASH, WASH + T_FIT
    Xt = torch.as_tensor(X, dtype=torch.float64)
    A = torch.cat([Xt, torch.ones(T, 1, dtype=torch.float64)], 1)
    Y = torch.as_tensor(np.stack(list(targets.values()), 1), dtype=torch.float64)
    W = ridge(Xt[a:b], Y[a:b])                                # toutes les cibles d'un coup
    P = A[b:] @ W
    Yt = Y[b:]
    nmse = ((P - Yt) ** 2).mean(0) / Yt.var(0, unbiased=False)
    C = np.clip(1 - nmse.numpy(), 0, 1)
    return dict(zip(targets, C.tolist()))


def summarize(C):
    lin = np.array([C[f"lin{k}"] for k in range(1, K_LIN + 1)])
    sq = sum(C[f"sq{k}"] for k in range(0, K_SQ + 1))
    cross = sum(v for k, v in C.items() if k.startswith("x"))
    return {"MC_k": lin.tolist(), "MC": float(lin.sum()), "C_k15": float(lin[14]), "nonlin2": float(sq + cross)}


def job(args):
    """Un modèle (entraîné ou non) sur une tâche et une seed → capacités. Tourne dans un processus séparé."""
    import torch
    torch.set_num_threads(1)
    from hfrnn import torch_models as TM
    label, kw, task, seed, n, steps = args
    kw = dict(kw)
    sparsity = kw.pop("sparsity", core.SPARSITY)
    Win, W = TM.make_reservoir(n, rho=core.RHO, in_scale=core.IN_SCALE, sparsity=sparsity, seed=seed)
    t0 = time.time()
    if kw["kind"] == "reservoir":
        model, nmse = TM.build_model(Win, W, "reservoir", seed, torch.float64, "cpu"), None
    else:
        u, y = tasks.load(task)
        nmse, info = TM.train_regression(u, y, Win, W, seed, steps=steps, dtype=torch.float64, return_model=True, **kw)
        model = info["model"]
    u = probe_input(seed)
    with torch.no_grad():
        X = model(torch.as_tensor(u, dtype=torch.float64).view(1, -1, 1))[0].numpy()
    out = summarize(capacities(X, u))
    out["nmse_task"] = nmse
    return label, task, seed, out, time.time() - t0


def run(a, path, res):
    jobs = []
    for seed in range(a.seeds):
        for l, kw, _ in UNTRAINED:
            if str(seed) not in res["results"].get("non entraîné", {}).get(l, {}):
                jobs.append((l, kw, "non entraîné", seed, a.n, a.steps))
        for t in TASKS:
            for l, kw, _ in TRAINED:
                if str(seed) not in res["results"].get(t, {}).get(l, {}):
                    jobs.append((l, kw, t, seed, a.n, a.steps))
    print(f"N = {a.n} | {a.seeds} seeds | {len(jobs)} mesures à faire | {a.jobs} processus", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, (l, t, seed, out, sec) in enumerate(ex.map(job, jobs), 1):
            res["results"].setdefault(t, {}).setdefault(l, {})[str(seed)] = out
            nm = f"NMSE tâche {out['nmse_task']:.3g}" if out["nmse_task"] is not None else ""
            print(f"[{(time.time() - t0) / 60:5.1f} min, {i}/{len(jobs)}] {t[:20]:20s} {l:34s} seed {seed}  "
                  f"MC = {out['MC']:5.1f}  C₁₅ = {out['C_k15']:.2f}  non-linéaire = {out['nonlin2']:5.1f}  {nm}",
                  flush=True)
            with open(path, "w") as f:
                json.dump(res, f, indent=1, ensure_ascii=False)


def mean_se(v):
    v = np.asarray(v, float)
    return v.mean(0), (v.std(0, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.zeros_like(v.mean(0)))


def report(res):
    R = res["results"]
    n = res["args"]["n"]
    base = R["non entraîné"]
    summary = {}
    print(f"\nCapacité mémoire linéaire MC (sur {n} au plus), capacité au délai 15, capacité non linéaire de degré 2 :")
    for l in base:
        mc, _ = mean_se([v["MC"] for v in base[l].values()])
        c15, _ = mean_se([v["C_k15"] for v in base[l].values()])
        nl, _ = mean_se([v["nonlin2"] for v in base[l].values()])
        print(f"  {'(non entraîné)':20s} {l:34s} MC = {mc:5.1f}  C₁₅ = {c15:.2f}  non-linéaire = {nl:5.1f}")
        summary.setdefault("non entraîné", {})[l] = {"MC": mc, "C15": c15, "nonlin2": nl}
    for t in TASKS:
        print(f"  après entraînement sur {t} :")
        for l, _, _ in TRAINED:
            if l not in R.get(t, {}):
                continue
            vals = list(R[t][l].values())
            mc, mcs = mean_se([v["MC"] for v in vals])
            c15, _ = mean_se([v["C_k15"] for v in vals])
            nl, nls = mean_se([v["nonlin2"] for v in vals])
            print(f"    {l:34s} MC = {mc:5.1f} ± {mcs:.1f}  C₁₅ = {c15:.2f}  non-linéaire = {nl:5.1f} ± {nls:.1f}  "
                  f"NMSE tâche = {np.exp(np.mean(np.log([v['nmse_task'] for v in vals]))):.3g}")
            summary.setdefault(t, {})[l] = {"MC": float(mc), "C15": float(c15), "nonlin2": float(nl)}
    with open(core.out_path("data", "memory_capacity_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    fig, axes = plt.subplots(len(TASKS), 2, figsize=(17, 4.3 * len(TASKS)), gridspec_kw={"width_ratios": [1.6, 1]})
    k = np.arange(1, K_LIN + 1)
    for i, t in enumerate(TASKS):
        ax = axes[i, 0]
        for l in base:
            m, _ = mean_se([v["MC_k"] for v in base[l].values()])
            ax.plot(k, m, color=COLORS[l], ls="--", lw=1.4, label=f"{l} (non entraîné)")
        for l, _, c in TRAINED:
            if l in R.get(t, {}):
                m, _ = mean_se([v["MC_k"] for v in R[t][l].values()])
                ax.plot(k, m, color=c, lw=2.2 if l == "réservoir accordé" else 1.4, label=l)
        ax.axvline(15, color="black", ls=":", lw=1)
        ax.set_xlim(0, 50); ax.set_ylim(-0.02, 1.02); ax.set_xlabel("délai k (pas)")
        ax.set_ylabel("capacité C_k"); ax.set_title(f"mémoire linéaire après entraînement sur {t}", fontsize=10)
        if i == 0:
            ax.legend(fontsize=7, ncol=2, loc="lower left")
        ax = axes[i, 1]
        labs = list(base) + [l for l, _, _ in TRAINED if l in R.get(t, {})]
        src = lambda l: base[l] if l in base else R[t][l]         # noqa: E731
        mc = [mean_se([v["MC"] for v in src(l).values()]) for l in labs]
        nl = [mean_se([v["nonlin2"] for v in src(l).values()]) for l in labs]
        y = np.arange(len(labs))
        ax.barh(y - 0.2, [m for m, _ in mc], 0.4, xerr=[s for _, s in mc], color=[COLORS[l] for l in labs],
                label="mémoire linéaire MC")
        ax.barh(y + 0.2, [m for m, _ in nl], 0.4, xerr=[s for _, s in nl], color=[COLORS[l] for l in labs],
                hatch="///", alpha=0.6, label="capacité non linéaire (degré 2)")
        ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8); ax.invert_yaxis()
        ax.set_xlabel(f"capacité (somme, au plus N = {n} au total)")
        if i == 0:
            ax.legend(fontsize=7, loc="lower right")
    fig.suptitle(f"Capacité mémoire (Jaeger) et capacité non linéaire, {n} neurones, "
                 f"{len(next(iter(base.values())))} seeds. Pointillés verticaux : délai 15 de la tâche mémoire.",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(core.out_path("figures", "memory_capacity.png"), dpi=110)
    print("\nFigure : results/figures/memory_capacity.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=core.STEPS)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 1)))
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    path = core.out_path("data", "memory_capacity.json")
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
