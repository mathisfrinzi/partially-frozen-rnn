"""
Partie 9, étape 3 — apprendre la décomposition au lieu de la choisir.

Les structures des parties 3 à 6 (accordé, tourné, cisaillé) sont des sous-espaces de matrices fixés d'avance. Ici le
sous-espace est tiré du gradient mesuré sur les données de la tâche, au réservoir de départ. Pour une dimension r
(100, 300, 1000 ; 1000 = autant de coefficients que la bande k = 10 à N = 100), W = W₀ + (r directions apprises) et les
biais et W_in sont appris comme dans les autres structures.

    directions principales    les r premières directions principales (non centrées) des gradients de 1 280 fenêtres
                              d'entraînement (« la descente se fait dans un petit sous-espace », Gur-Ari et al. 2018)
    directions à accord       parmi les directions principales, les r dont la composante moyenne a le même signe et
                              est grande sur 1 280 fenêtres d'entraînement et sur 256 fenêtres de validation (produit
                              des deux moyennes) : la direction stable d'un lot à l'autre, pas seulement la plus variable
    positions du gradient     les r coefficients de W dont le gradient moyen est le plus grand (masque de Fisher,
                              Sung et al. 2021)
    positions à accord        les r coefficients dont le gradient moyen entraînement × validation est le plus grand
    contrôle : directions aléatoires   sous-espace orthonormé tiré au hasard (dimension intrinsèque, Li et al. 2018)
    contrôle : positions au hasard     r coefficients tirés au hasard (le contrôle « appris dans W₀ » de la partie 5)

Références (mêmes graines, mêmes tâches, parties 3 à 6) : RNN, accordé, cisaillé en bande k = 10, biais et entrées seuls.

    python experiments/46_sous_espace_appris.py --seed 0     # une graine (~35 min CPU), écrit sous_espace_s0.json
    python experiments/46_sous_espace_appris.py --report
"""
import argparse
import glob
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn import core

TASKS = ["Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)", "Lorenz (+10)", "Hénon (+3)"]
DIMS = [100, 300, 1000]
METHODS = ["directions principales", "directions à accord", "positions du gradient", "positions à accord",
           "contrôle : directions aléatoires", "contrôle : positions au hasard"]
REFS = {"RNN": "RNN", "réservoir accordé (valeurs singulières)": "accordé", "réservoir cisaillé en bande, k = 10": "cisaillé k = 10",
        "réservoir à biais et entrées appris (sans g)": "biais et entrées", "réservoir classique": "classique"}
N_A, N_B = 1280, 256


def data(name):
    spec = importlib.util.spec_from_file_location("dec36", Path(__file__).with_name("36_decompositions.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.data(name)


def references():
    """log10 NMSE de test des références, {tâche: {étiquette: {graine: valeur}}}, parties 3 à 6 à 100 neurones."""
    out = {}
    for f in sorted(glob.glob(str(core.out_path("data", "decompositions_n100*.json")))):
        if f.endswith("_summary.json"):
            continue
        for t, R in json.load(open(f))["results"].items():
            for lab, cells in R.items():
                if lab in REFS:
                    for s, v in cells.items():
                        out.setdefault(t, {}).setdefault(lab, {})[s] = float(np.log10(v["nmse"]))
    return out


# --------------------------------------------------------------------------------------------- construction des sous-espaces
def gradient_sets(u, y, nt, Win, W, seed, nA, nB):
    """Gradients par fenêtre (partie W seulement) : A (entraînement, premiers 75 %) et B (validation, derniers 25 %)."""
    from hfrnn import structure_signals as SS
    w, c = SS.ridge_readout(u, y, nt, Win, W)
    nv = int(0.75 * nt)
    rng = np.random.default_rng(700 + seed)
    a = rng.integers(0, nv - core.WIN_LEN - 1, nA)
    b = rng.integers(nv, nt - core.WIN_LEN - 1, nB)
    return SS.window_grads(u, y, a, Win, W, w, c)[0], SS.window_grads(u, y, b, Win, W, w, c)[0]


def subspaces(GA, GB, dims, seed):
    """{(méthode, r): {"V": ... } ou {"idx": ...}} à partir des gradients A et B."""
    D = GA.shape[1]
    _, _, Vt = np.linalg.svd(GA, full_matrices=False)               # directions principales non centrées
    mA, mB = GA.mean(0), GB.mean(0)
    zA, zB = (GA @ Vt.T).mean(0), (GB @ Vt.T).mean(0)
    rng = np.random.default_rng(800 + seed)
    out = {}
    for r in dims:
        r = min(r, Vt.shape[0], D)
        out[("directions principales", r)] = {"V": Vt[:r]}
        out[("directions à accord", r)] = {"V": Vt[np.sort(np.argsort(-(zA * zB))[:r])]}
        out[("positions du gradient", r)] = {"idx": np.sort(np.argsort(-np.abs(mA))[:r])}
        out[("positions à accord", r)] = {"idx": np.sort(np.argsort(-(mA * mB))[:r])}
        Q, _ = np.linalg.qr(rng.normal(size=(D, r)))
        out[("contrôle : directions aléatoires", r)] = {"V": Q.T}
        out[("contrôle : positions au hasard", r)] = {"idx": np.sort(rng.permutation(D)[:r])}
    return out


def run(a, path):
    import torch
    from hfrnn import structure_signals as SS
    from hfrnn import torch_models as TM
    torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    res = json.load(open(path)) if path.exists() else {"args": {"n": a.n, "steps": a.steps}, "results": {}}
    t0 = time.time()
    for task in (a.tasks or TASKS):
        u, y, nt = data(task)
        d = 1 if u.ndim == 1 else u.shape[1]
        Win, W = SS.reservoir(a.n, d, a.seed)
        cell = res["results"].setdefault(task, {})
        todo = [(m, r) for r in a.dims for m in METHODS if f"{m} | {r}" not in cell]
        if not todo:
            continue
        GA, GB = gradient_sets(u, y, nt, Win, W, a.seed, a.n_a, a.n_b)
        sub = subspaces(GA, GB, a.dims, a.seed)
        del GA, GB
        for (m, r) in todo:
            v, info = TM.train_regression(u, y, Win, W, a.seed, kind="subspace", steps=a.steps, device=device, n_train=nt,
                                          **sub[(m, min(r, max(k[1] for k in sub)))])
            cell[f"{m} | {r}"] = {"log10": float(np.log10(v)), "n_params": info["n_params"]}
            json.dump(res, open(path, "w"), indent=1, ensure_ascii=False)
        print(f"[{(time.time() - t0) / 60:5.1f} min] {task:20s} graine {a.seed} : {len(todo)} sous-espaces entraînés", flush=True)


# ------------------------------------------------------------------------------------------------ rapport
def load(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        seed = Path(f).stem.rsplit("_s", 1)[1]
        for task, cell in json.load(open(f))["results"].items():
            for key, v in cell.items():
                out.setdefault(task, {}).setdefault(key, {})[seed] = v["log10"]
    return out


def report(a):
    R = load(str(core.out_path("data", "sous_espace_s[0-9]*.json")))
    ref = references()
    tasks = [t for t in TASKS if t in R]
    dims = sorted({int(k.split(" | ")[1]) for t in tasks for k in R[t]})
    seeds = sorted({s for t in tasks for k in R[t] for s in R[t][k]})
    print(f"\n{len(tasks)} tâches, graines {seeds}. log10 NMSE de test, moyenne sur les graines (plus bas = mieux).")
    m_ = lambda d, ks: np.mean([d[k] for k in ks if k in d])   # noqa: E731
    for t in tasks:
        print(f"\n{t}")
        print(f"  {'':36s}" + "".join(f"{'r = ' + str(r):>10s}" for r in dims))
        for m in METHODS:
            print(f"  {m:36s}" + "".join(f"{m_(R[t].get(f'{m} | {r}', {}), seeds) if R[t].get(f'{m} | {r}') else float('nan'):>10.2f}"
                                          for r in dims))
        print("  références : " + " ; ".join(f"{REFS[l]} {m_(ref[t][l], seeds):.2f}" for l in REFS if l in ref.get(t, {})))
    # écarts pairés : méthode moins contrôle aléatoire de même dimension (directions pour les directions, positions pour les positions)
    print("\nÉcart moyen au contrôle aléatoire de même type et de même dimension (log10 NMSE, négatif = mieux), "
          "moyenne sur les tâches [nombre de tâches où la méthode gagne]")
    summary = {}
    ctrl = {"directions principales": "contrôle : directions aléatoires", "directions à accord": "contrôle : directions aléatoires",
            "positions du gradient": "contrôle : positions au hasard", "positions à accord": "contrôle : positions au hasard"}
    print(f"  {'':28s}" + "".join(f"{'r = ' + str(r):>16s}" for r in dims))
    for m, c in ctrl.items():
        row = []
        for r in dims:
            diffs = [m_(R[t][f"{m} | {r}"], seeds) - m_(R[t][f"{c} | {r}"], seeds) for t in tasks
                     if f"{m} | {r}" in R[t] and f"{c} | {r}" in R[t]]
            row.append((float(np.mean(diffs)), int(sum(d < 0 for d in diffs)), len(diffs)) if diffs else None)
        summary[m] = row
        print(f"  {m:28s}" + "".join(f"{x[0]:>+10.2f} [{x[1]}/{x[2]}]" if x else f"{'':>16s}" for x in row))
    print("\nÉcart à la bande k = 10 (même nombre de coefficients à r = 1000) :")
    band = "réservoir cisaillé en bande, k = 10"
    for m in METHODS:
        if 1000 in dims:
            diffs = [m_(R[t][f"{m} | 1000"], seeds) - m_(ref[t][band], seeds) for t in tasks
                     if f"{m} | 1000" in R[t] and band in ref.get(t, {})]
            if diffs:
                print(f"  {m:36s} {np.mean(diffs):>+6.2f}  [{sum(d < 0 for d in diffs)}/{len(diffs)} tâches mieux que la bande]")
    json.dump({"écart au contrôle": summary}, open(core.out_path("data", "sous_espace_summary.json"), "w"), indent=1,
              ensure_ascii=False)
    figure(R, ref, tasks, dims, seeds)


def figure(R, ref, tasks, dims, seeds):
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.6 * len(tasks), 3.6), squeeze=False, sharex=True)
    for ax, t in zip(axes[0], tasks):
        for m in METHODS:
            ys = [np.mean(list(R[t][f"{m} | {r}"].values())) if f"{m} | {r}" in R[t] else np.nan for r in dims]
            ax.plot(dims, ys, marker="o", ls="--" if m.startswith("contrôle") else "-", label=m, lw=1.4)
        for lab, col in (("réservoir cisaillé en bande, k = 10", "k"), ("RNN", "gray"), ("réservoir accordé (valeurs singulières)", "tab:brown")):
            if lab in ref.get(t, {}):
                ax.axhline(np.mean(list(ref[t][lab].values())), color=col, lw=0.8, ls=":")
                ax.annotate(REFS[lab], (dims[0], np.mean(list(ref[t][lab].values()))), fontsize=6, color=col)
        ax.set_xscale("log"); ax.set_title(t, fontsize=9); ax.set_xlabel("dimension r du sous-espace")
    axes[0, 0].set_ylabel("log10 NMSE de test"); axes[0, -1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(core.out_path("figures", "sous_espace_appris.png"), dpi=110)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--steps", type=int, default=core.STEPS)
    p.add_argument("--dims", type=int, nargs="*", default=DIMS)
    p.add_argument("--n_a", type=int, default=N_A)
    p.add_argument("--n_b", type=int, default=N_B)
    p.add_argument("--tasks", nargs="*")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--quick", action="store_true", help="vérification rapide (16 neurones, quelques pas)")
    a = p.parse_args()
    if a.quick:
        a.n, a.steps, a.dims, a.n_a, a.n_b = 16, 6, [8, 30], 48, 16
    if not a.report:
        run(a, core.out_path("data", f"sous_espace_s{a.seed}.json"))
    else:
        report(a)
