"""
Partie 4.8 — Un réservoir, plusieurs tâches.

Un seul réservoir aléatoire W₀ sert à toutes les tâches ; seul le réglage g (une amplitude par direction) change
d'une tâche à l'autre, comme on change de station sur un même poste de radio. Une tâche tient alors en quelques
centaines de nombres, et en apprendre une nouvelle ne peut pas abîmer les anciennes : leurs réglages sont rangés à
part. On compare :

    accordé, tout par tâche            g, entrées, biais et lecture appris pour chaque tâche (la partie 3)
    accordé, entrées partagées         entrées W_in figées, communes à toutes les tâches ; par tâche : g, biais,
                                       lecture (≈ 3 N nombres)
    RNN, une copie par tâche           un RNN complet par tâche (≈ N² nombres par tâche)
    RNN séquentiel                     un seul RNN qui apprend Mackey-Glass, puis NARMA10, puis la mémoire
    accordé séquentiel, un réglage     un seul réservoir accordé, un seul g, appris dans le même ordre : sépare l'effet
                                       « peu de paramètres » de l'effet « réglages rangés à part »

Pour les deux versions séquentielles, chaque tâche est mesurée à la fin de son apprentissage puis à la toute fin, avec
la lecture apprise à son tour (ce que l'on déploierait). On mesure aussi les réglages croisés : le g appris pour une
tâche, utilisé sur une autre (lecture réapprise), et la moyenne de deux réglages.

    python experiments/25_multitask.py              # N = 100, 3 seeds, CPU (~10 min sur 2 cœurs)
    python experiments/25_multitask.py --n 500      # GPU conseillé
    python experiments/25_multitask.py --report
Sorties : results/data/multitask_n{N}.json, results/figures/multitask_n{N}.png
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

TASKS = ["Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)"]
SHORT = {"Mackey-Glass (+15)": "Mackey-Glass", "NARMA10": "NARMA10", "Mémoire (délai 15)": "Mémoire"}
CONDITIONS = [
    ("accordé, tout par tâche", "tab:blue"),
    ("accordé, entrées partagées", "navy"),
    ("RNN, une copie par tâche", "tab:red"),
    ("RNN séquentiel", "salmon"),
    ("accordé séquentiel, un réglage", "lightsteelblue"),
]
RNN = {"kind": "convex", "alpha": 1.0}
TUNED = {"kind": "spectrum"}


def readout(model, ut, yt):
    """Lecture ridge apprise sur la partie entraînement (comme regression_nmse), renvoyée pour être réutilisée."""
    import torch
    from hfrnn.torch_models import ridge
    NT, WO = core.n_train(ut), core.WASHOUT
    with torch.no_grad():
        X = model(ut.view(1, -1, 1))[0]
    return ridge(X[WO:NT], yt[WO:NT])


def nmse_with(model, ut, yt, w):
    """NMSE test avec une lecture donnée (celle apprise au moment où la tâche a été apprise)."""
    import torch
    NT = core.n_train(ut)
    with torch.no_grad():
        X = model(ut.view(1, -1, 1))[0].double()
    pred = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype, device=X.device)], 1) @ w
    yd = yt.double()
    return float(((pred[NT:] - yd[NT:]) ** 2).mean() / yd[NT:].var(unbiased=False))


def one_seed(args):
    import torch
    seed, n, steps, dev = args
    if dev == "cpu":
        torch.set_num_threads(1)
    from hfrnn import torch_models as TM
    dt = torch.float64 if dev == "cpu" else torch.float32
    Win, W = TM.make_reservoir(n, rho=core.RHO, in_scale=core.IN_SCALE, sparsity=core.SPARSITY, seed=seed)
    data = {t: tasks.load(t) for t in TASKS}
    T = {t: (torch.as_tensor(u, dtype=dt, device=dev), torch.as_tensor(y, dtype=dt, device=dev))
         for t, (u, y) in data.items()}
    tr = lambda t, **kw: TM.train_regression(*data[t], Win, W, seed, steps=steps, device=dev, dtype=dt,  # noqa: E731
                                             return_model=True, **kw)
    out, stored = {}, {}
    t0 = time.time()
    # 1. séparés : un modèle par tâche
    for cond, kw in (("accordé, tout par tâche", TUNED), ("accordé, entrées partagées", dict(TUNED, learn_win=False)),
                     ("RNN, une copie par tâche", RNN)):
        for t in TASKS:
            s, info = tr(t, **kw)
            out.setdefault(cond, {})[t] = {"final": s, "n_par_tache": info["n_params"] + n + 1}
            if cond == "accordé, entrées partagées":
                m = info["model"]
                stored[t] = (m.g.detach().clone(), m.bias.detach().clone())
    # 2. séquentiels : un seul modèle, les tâches l'une après l'autre
    for cond, kw in (("RNN séquentiel", RNN), ("accordé séquentiel, un réglage", TUNED)):
        model, ws, res = None, {}, {}
        for i, t in enumerate(TASKS):
            s, info = tr(t, model=model, **kw)
            model = info["model"]
            ws[t] = readout(model, *T[t])
            res[t] = {"juste après": s, "n_par_tache": info["n_params"] + n + 1}
            for prev in TASKS[:i + 1]:                       # suivi de toutes les tâches déjà apprises
                res[prev].setdefault("après chaque étape", []).append(nmse_with(model, *T[prev], ws[prev]))
        for t in TASKS:
            res[t]["final"] = res[t]["après chaque étape"][-1]
            res[t]["final, lecture réapprise"] = TM.regression_nmse(model, *T[t])
        out[cond] = res
    # 3. réglages croisés (entrées partagées) : g de la tâche a, évalué sur la tâche b (lecture réapprise)
    probe = TM.build_model(Win, W, "spectrum", seed, dt, dev, learn_win=False)
    regl = dict(stored)
    regl["moyenne Mackey-Glass + NARMA10"] = tuple((stored[TASKS[0]][k] + stored[TASKS[1]][k]) / 2 for k in (0, 1))
    regl["réservoir de départ (g = 1)"] = (torch.ones_like(stored[TASKS[0]][0]), stored[TASKS[0]][1] * 0)
    cross = {}
    with torch.no_grad():
        for name, (g, b) in regl.items():
            probe.g.copy_(g); probe.bias.copy_(b)
            cross[name] = {t: TM.regression_nmse(probe, *T[t]) for t in TASKS}
    out["réglages croisés"] = cross
    # contrôle : remettre le réglage rangé de chaque tâche dans un même modèle redonne exactement son score
    with torch.no_grad():
        check = {}
        for t in TASKS:
            probe.g.copy_(stored[t][0]); probe.bias.copy_(stored[t][1])
            check[t] = TM.regression_nmse(probe, *T[t])
    out["contrôle, réglage rangé remis en place"] = check
    return seed, out, time.time() - t0


def run(a, path, res):
    import torch
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    todo = [s for s in range(a.seeds) if str(s) not in res["results"]]
    print(f"N = {a.n} | {a.steps} pas | seeds à faire : {todo} | appareil {dev}", flush=True)
    jobs = [(s, a.n, a.steps, dev) for s in todo]
    workers = 1 if dev != "cpu" else max(1, min(a.jobs, len(jobs)))
    if workers == 1:                         # GPU : tout dans ce processus (CUDA ne redémarre pas dans un sous-processus)
        results = map(one_seed, jobs)
    else:
        ex = ProcessPoolExecutor(workers)
        results = ex.map(one_seed, jobs)
    for seed, out, sec in results:
        res["results"][str(seed)] = out
        print(f"seed {seed} terminée en {sec / 60:.1f} min", flush=True)
        with open(path, "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)


def gm(v):
    return float(np.exp(np.mean(np.log(v))))


def report(res, n):
    R = res["results"]
    seeds = sorted(R)
    summary = {}
    print(f"\nN = {n}, {len(seeds)} seeds. NMSE test (moyenne géométrique) ; nombres à stocker par tâche")
    for cond, _ in CONDITIONS:
        row = {t: gm([R[s][cond][t]["final"] for s in seeds]) for t in TASKS}
        store = R[seeds[0]][cond][TASKS[0]]["n_par_tache"]
        summary[cond] = {"nmse": row, "nombres par tâche": store}
        print(f"  {cond:34s} " + "  ".join(f"{SHORT[t]} {row[t]:.3g}" for t in TASKS) + f"   ({store} nombres)")
    for cond in ("RNN séquentiel", "accordé séquentiel, un réglage"):
        print(f"  {cond} : chaque tâche juste après son apprentissage → à la fin (lecture d'origine ; lecture réapprise)")
        for t in TASKS:
            a = gm([R[s][cond][t]["juste après"] for s in seeds]); b = gm([R[s][cond][t]["final"] for s in seeds])
            c = gm([R[s][cond][t]["final, lecture réapprise"] for s in seeds])
            print(f"    {SHORT[t]:13s} {a:.3g} → {b:.3g}  (×{b / a:.1f} plus d'erreur ; lecture réapprise : {c:.3g})")
            summary.setdefault("oubli", {}).setdefault(cond, {})[t] = {"juste après": a, "fin": b, "fin, lecture réapprise": c}
    sep = gm([R[s]["accordé, entrées partagées"][t]["final"] for s in seeds for t in TASKS])
    chk = gm([R[s]["contrôle, réglage rangé remis en place"][t] for s in seeds for t in TASKS])
    print(f"  contrôle : réglages rangés remis dans un même modèle {chk:.4g} contre {sep:.4g} (identiques attendus)")
    cross = {k: {t: gm([R[s]["réglages croisés"][k][t] for s in seeds]) for t in TASKS}
             for k in R[seeds[0]]["réglages croisés"]}
    summary["réglages croisés"] = cross
    print("  réglages croisés (lecture réapprise), lignes = réglage utilisé, colonnes = tâche :")
    for k, row in cross.items():
        print(f"    {k:34s} " + "  ".join(f"{SHORT[t]} {row[t]:.3g}" for t in TASKS))
    with open(core.out_path("data", f"multitask_n{n}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.6), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    ax = axes[0]
    w = 0.8 / len(CONDITIONS)
    for i, (cond, c) in enumerate(CONDITIONS):
        v = [summary[cond]["nmse"][t] for t in TASKS]
        k = summary[cond]["nombres par tâche"]
        kk = f"{k:,}".replace(",", " ")
        what = (f"un seul modèle ({k - n - 1:,} nombres) pour toutes les tâches".replace(",", " ")
                if "séquentiel" in cond else f"{kk} nombres par tâche")
        ax.bar(np.arange(3) + (i - 2) * w, v, w, color=c, label=f"{cond} ({what})")
    ax.set_xticks(range(3)); ax.set_xticklabels([SHORT[t] for t in TASKS]); ax.set_yscale("log")
    ax.set_ylabel("NMSE test (log)"); ax.legend(fontsize=7); ax.set_title("à la fin : toutes les tâches apprises")
    ax = axes[1]
    for cond, c, ls in (("RNN séquentiel", "tab:red", "-"), ("accordé séquentiel, un réglage", "tab:blue", "--")):
        for j, t in enumerate(TASKS):
            curve = np.exp(np.mean([np.log(R[s][cond][t]["après chaque étape"]) for s in seeds], 0))
            x = np.arange(j, 3)
            ax.plot(x, curve, ls=ls, marker="o", color=c, alpha=[1, 0.7, 0.45][j],
                    label=f"{cond} — {SHORT[t]}")
    for j, t in enumerate(TASKS):
        v = summary["accordé, tout par tâche"]["nmse"][t]
        ax.plot([j, 2], [v, v], color="tab:blue", lw=4, alpha=0.3,
                label="accordé, réglages rangés par tâche : rien ne bouge" if j == 0 else None)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"après {SHORT[t]}" for t in TASKS]); ax.set_yscale("log")
    ax.set_ylabel("NMSE test, lecture d'origine (log)"); ax.legend(fontsize=6.5)
    ax.set_title("apprendre l'une après l'autre : qui oublie ?")
    ax = axes[2]
    names = list(cross)
    M = np.array([[np.log10(cross[k][t] / summary["accordé, entrées partagées"]["nmse"][t]) for t in TASKS] for k in names])
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=1.5)
    for i in range(len(names)):
        for j in range(3):
            ax.text(j, i, f"×{10 ** M[i, j]:.1f}" if M[i, j] < 2 else f"×{10 ** M[i, j]:.0f}", ha="center", va="center",
                    fontsize=8)
    ax.set_xticks(range(3)); ax.set_xticklabels([SHORT[t] for t in TASKS])
    ax.set_yticks(range(len(names))); ax.set_yticklabels([f"g de {SHORT.get(k, k)}" if k in TASKS else k for k in names],
                                                         fontsize=8)
    ax.set_title("réglages croisés (entrées partagées) :\nerreur relative au bon réglage", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, label="log10(erreur / erreur avec le bon réglage)")
    fig.suptitle(f"Partie 7.3 — un réservoir, plusieurs tâches ({n} neurones, {len(seeds)} seeds)", fontsize=12)
    fig.tight_layout(); fig.savefig(core.out_path("figures", f"multitask_n{n}.png"), dpi=110)
    print(f"\nFigure : results/figures/multitask_n{n}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=core.STEPS)
    ap.add_argument("--jobs", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--cpu", action="store_true", help="forcer le CPU (float64)")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    path = core.out_path("data", f"multitask_n{a.n}.json")
    res = {"args": {"n": a.n, "steps": a.steps}, "results": {}}
    if path.exists():
        old = json.load(open(path))
        if old["args"] == res["args"]:
            res = old
    if not a.report:
        run(a, path, res)
    report(res, a.n)


if __name__ == "__main__":
    main()
