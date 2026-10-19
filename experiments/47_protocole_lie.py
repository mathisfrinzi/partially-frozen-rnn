"""
Partie 9, étape 4 — le protocole « PCA du gradient tangent » tel qu'il a été formulé.

    1. Sondage : depuis le réservoir de départ, on échantillonne des lots et on calcule à chaque fois le vecteur tangent
       X_k = M⁻¹ G_k, où G_k = ∂L_k/∂W est le gradient de la perte de la fenêtre k et M = W₀ la matrice de départ
       (flot de gradient multiplicatif : W = W₀·exp(X), donc ΔW = W₀·X au premier ordre).
    2. Covariance des tangents C = E[X ⊗ X] (non centrée) sur 2 560 fenêtres.
    3. Diagonalisation : les directions propres, ordonnées par valeur propre (raideur), donnent un découpage de gl(n) appris.

M est presque singulière (plus petite valeur singulière de l'ordre de 10⁻⁴ à N = 100) : M⁻¹ est régularisée,
X = (MᵀM + λI)⁻¹ Mᵀ G avec λ = (ε·σ_max)². Trois définitions du tangent sont comparées : le gradient brut G (comme au bloc 11,
mais avec deux fois plus de fenêtres), et M⁻¹G régularisé avec ε = 0,01 et ε = 0,1.

    Test statique (dimension d = 100, 300, 1000, même nombre de coefficients appris pour toutes les lignes)
        premières       les d premières directions propres (les plus raides)
        suivantes       les d directions propres de rangs 1000 à 1000 + d (la suite du spectre estimé)
        aléatoires      d directions orthonormées tirées au hasard (même transformation W₀·X)
    Test du dégel (hiérarchie des valeurs propres comme ordre d'apprentissage) : les 1000 premières directions, en dix blocs
    de 100, sont dégelées l'une après l'autre (un bloc tous les ~136 pas), les autres coefficients restant à zéro jusque-là.
        valeurs propres décroissantes     le bloc le plus raide d'abord
        ordre aléatoire                    les mêmes blocs dans un ordre tiré au hasard
        valeurs propres croissantes        le bloc le moins raide d'abord
    À la fin, les 1000 coefficients sont tous libres : seule l'histoire de l'entraînement diffère.

Références : RNN, accordé, cisaillé k = 10, biais et entrées seuls (parties 3 à 6, mêmes graines).

    python experiments/47_protocole_lie.py --seed 0 --threads 1     # ~100 min, écrit protocole_lie_s0.json
    python experiments/47_protocole_lie.py --report
"""
import argparse
import gc
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
EPS = [0.01, 0.1]
TANGENTS = ["G brut"] + [f"M⁻¹G (ε = {e})" for e in EPS]
MODES = ["premières", "suivantes", "aléatoires"]
ORDERS = ["valeurs propres décroissantes", "ordre aléatoire", "valeurs propres croissantes"]
N_A, N_BLOCKS = 2560, 10


def _load(name, tag):
    spec = importlib.util.spec_from_file_location(tag, Path(__file__).with_name(name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def keys(dims):
    return ([f"{t} | {m} | {d}" for t in TANGENTS for m in MODES for d in dims]
            + [f"{t} | dégel {o} | {max(dims)}" for t in TANGENTS for o in ORDERS])


def tangent_samples(G, W0, name):
    """Vecteurs tangents (n, N²) à partir des gradients par fenêtre G (n, N²)."""
    if name == "G brut":
        return G
    n, N = len(G), W0.shape[0]
    eps = float(name.split("= ")[1].rstrip(")"))
    lam = (eps * np.linalg.svd(W0, compute_uv=False)[0]) ** 2
    P = np.linalg.solve(W0.T @ W0 + lam * np.eye(N), W0.T)             # (MᵀM + λI)⁻¹ Mᵀ
    return np.matmul(P, G.reshape(n, N, N)).reshape(n, -1)


def to_weights(V, W0, name):
    """Directions de l'algèbre (r, N²) -> directions de matrices ΔW : identité pour le brut, W₀·X pour le multiplicatif."""
    if name == "G brut":
        return V
    N = W0.shape[0]
    return np.matmul(W0, V.reshape(len(V), N, N)).reshape(len(V), -1)


def run(a, path):
    import torch
    from hfrnn import structure_signals as SS
    from hfrnn import torch_models as TM
    torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    res = json.load(open(path)) if path.exists() else {"args": {"n": a.n, "steps": a.steps}, "results": {}, "spectres": {}}
    m46 = _load("46_sous_espace_appris.py", "m46")
    t0 = time.time()
    bs = max(1, max(a.dims) // N_BLOCKS)
    R = bs * N_BLOCKS                                                  # directions du dégel
    for task in (a.tasks or TASKS):
        u, y, nt = m46.data(task)
        d = 1 if u.ndim == 1 else u.shape[1]
        Win, W = SS.reservoir(a.n, d, a.seed)
        cell = res["results"].setdefault(task, {})
        todo = [k for k in keys(a.dims) if k not in cell]
        if not todo:
            continue
        w, c = SS.ridge_readout(u, y, nt, Win, W)
        rng = np.random.default_rng(900 + a.seed)
        starts = rng.integers(0, int(0.75 * nt) - core.WIN_LEN - 1, a.n_a)         # fenêtres d'entraînement seulement
        G = SS.window_grads(u, y, starts, Win, W, w, c)[0].astype(np.float32)
        print(f"[{(time.time() - t0) / 60:5.1f} min] {task:20s} graine {a.seed} : {len(G)} gradients", flush=True)
        for tname in TANGENTS:
            mine = [k for k in todo if k.startswith(tname + " |")]
            if not mine:
                continue
            X = tangent_samples(G, W, tname).astype(np.float32)
            _, s, Vt = np.linalg.svd(X, full_matrices=False)
            res["spectres"].setdefault(task, {})[tname] = [float(x) for x in (s[:200] ** 2 / len(X))]
            D = Vt.shape[1]
            qrng = np.random.default_rng(1000 + a.seed)
            runs = {}
            for dd in a.dims:
                runs[f"{tname} | premières | {dd}"] = (Vt[:dd], None)
                runs[f"{tname} | suivantes | {dd}"] = (Vt[max(a.dims):max(a.dims) + dd], None)
                runs[f"{tname} | aléatoires | {dd}"] = (np.linalg.qr(qrng.normal(size=(D, dd)))[0].T, None)
            pool = Vt[:R]
            gap = max(1, a.steps // (N_BLOCKS + 1))
            orders = {"valeurs propres décroissantes": np.arange(N_BLOCKS),
                      "ordre aléatoire": np.random.default_rng(1100 + a.seed).permutation(N_BLOCKS),
                      "valeurs propres croissantes": np.arange(N_BLOCKS)[::-1]}
            for o, ordre in orders.items():
                rank = np.empty(N_BLOCKS, int)
                rank[ordre] = np.arange(N_BLOCKS)                          # rang de dégel de chaque bloc
                runs[f"{tname} | dégel {o} | {max(a.dims)}"] = (pool, np.repeat(rank, bs) * gap)
            for k in mine:
                V, thaw = runs[k]
                kw = {"V": to_weights(V, W, tname)}
                if thaw is not None:
                    kw["thaw"] = thaw
                v, info = TM.train_regression(u, y, Win, W, a.seed, kind="subspace", steps=a.steps, device=device,
                                              n_train=nt, **kw)
                cell[k] = {"log10": float(np.log10(v)), "n_params": info["n_params"]}
                json.dump(res, open(path, "w"), indent=1, ensure_ascii=False)
            print(f"[{(time.time() - t0) / 60:5.1f} min] {task:20s} graine {a.seed} : {tname} ({len(mine)} entraînements)",
                  flush=True)
            del X, Vt
            gc.collect()


# ------------------------------------------------------------------------------------------------ rapport
def load(pattern):
    out, spec = {}, {}
    for f in sorted(glob.glob(pattern)):
        seed = Path(f).stem.rsplit("_s", 1)[1]
        r = json.load(open(f))
        for task, cell in r["results"].items():
            for k, v in cell.items():
                out.setdefault(task, {}).setdefault(k, {})[seed] = v["log10"]
        for task, T in r.get("spectres", {}).items():
            for t, ev in T.items():
                spec.setdefault(task, {}).setdefault(t, {})[seed] = ev
    return out, spec


def report(a):
    R, spec = load(str(core.out_path("data", "protocole_lie_s[0-9]*.json")))
    ref = _load("46_sous_espace_appris.py", "m46").references()
    tasks = [t for t in TASKS if t in R]
    dims = sorted({int(k.split(" | ")[2]) for t in tasks for k in R[t] if "dégel" not in k})
    seeds = sorted({s for t in tasks for k in R[t] for s in R[t][k]})
    left = [t for t in tasks if not all(k in R[t] and len(R[t][k]) == len(seeds) for k in keys(dims))]
    if left:                                                           # tâches incomplètes : laissées de côté
        print("tâches incomplètes, laissées de côté :", ", ".join(f"{t} ({sum(len(v) for v in R[t].values())} "
              f"entraînements sur {len(keys(dims)) * len(seeds)})" for t in left))
        tasks = [t for t in tasks if t not in left]
    mean = lambda t, k: float(np.mean(list(R[t][k].values()))) if k in R[t] else np.nan   # noqa: E731
    print(f"\n{len(tasks)} tâches, graines {seeds}. log10 NMSE de test, moyenne sur les graines (plus bas = mieux).")
    for t in tasks:
        print(f"\n{t}")
        for tg in TANGENTS:
            print(f"  [{tg}]")
            for m in MODES:
                print(f"    {m:30s}" + "".join(f"{mean(t, f'{tg} | {m} | {d}'):>9.2f}" for d in dims) + "   (d = " +
                      ", ".join(map(str, dims)) + ")")
            for o in ORDERS:
                print(f"    dégel {o:30s}{mean(t, f'{tg} | dégel {o} | {max(dims)}'):>9.2f}")
        print("  références : " + " ; ".join(f"{m46_short(l)} {np.mean(list(ref[t][l].values())):.2f}" for l in ref.get(t, {})
                                              if l in ("RNN", "réservoir cisaillé en bande, k = 10",
                                                       "réservoir accordé (valeurs singulières)",
                                                       "réservoir à biais et entrées appris (sans g)")))
    print("\nSpectre de C (valeurs propres, moyenne sur les graines) : rapport λ_10 / λ_1, λ_100 / λ_1 et nombre de directions "
          "pour 90 % de la trace (parmi les 200 premières) :")
    for t in tasks:
        for tg in TANGENTS:
            if tg in spec.get(t, {}):
                ev = np.mean(list(spec[t][tg].values()), 0)
                i10, i100 = min(9, len(ev) - 1), min(99, len(ev) - 1)
                print(f"  {t[:18]:18s} {tg:20s} {ev[i10] / ev[0]:.3f}  {ev[i100] / ev[0]:.4f}  "
                      f"{int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.9)) + 1}")
    print("\nÉcarts moyens (log10 NMSE, négatif = la première ligne est mieux), moyenne sur les tâches [victoires]")
    cmp_static = [("premières − aléatoires", "premières", "aléatoires"), ("premières − suivantes", "premières", "suivantes")]
    summary = {}
    for tg in TANGENTS:
        print(f"  [{tg}]")
        for name, A, B in cmp_static:
            row = []
            for d in dims:
                diffs = [mean(t, f"{tg} | {A} | {d}") - mean(t, f"{tg} | {B} | {d}") for t in tasks]
                diffs = [x for x in diffs if np.isfinite(x)]
                row.append(f"{np.mean(diffs):+.2f} [{sum(x < 0 for x in diffs)}/{len(diffs)}]" if diffs else "")
            print(f"    {name:34s}" + "".join(f"{x:>16s}" for x in row) + "   (d = " + ", ".join(map(str, dims)) + ")")
            summary[f"{tg} | {name}"] = row
        for name, A, B in (("dégel décroissant − aléatoire", ORDERS[0], ORDERS[1]),
                           ("dégel décroissant − croissant", ORDERS[0], ORDERS[2])):
            diffs = [mean(t, f"{tg} | dégel {A} | {max(dims)}") - mean(t, f"{tg} | dégel {B} | {max(dims)}") for t in tasks]
            diffs = [x for x in diffs if np.isfinite(x)]
            if diffs:
                print(f"    {name:34s}{np.mean(diffs):>+16.2f} [{sum(x < 0 for x in diffs)}/{len(diffs)}]")
                summary[f"{tg} | {name}"] = [float(np.mean(diffs))]
    json.dump(summary, open(core.out_path("data", "protocole_lie_summary.json"), "w"), indent=1, ensure_ascii=False)
    figure(R, ref, tasks, dims)


def m46_short(label):
    return {"RNN": "RNN", "réservoir cisaillé en bande, k = 10": "cisaillé k=10",
            "réservoir accordé (valeurs singulières)": "accordé", "réservoir à biais et entrées appris (sans g)": "biais et entrées"}[label]


def figure(R, ref, tasks, dims):
    fig, axes = plt.subplots(len(TANGENTS), len(tasks), figsize=(3.4 * len(tasks), 3.0 * len(TANGENTS)), squeeze=False,
                             sharex=True)
    for i, tg in enumerate(TANGENTS):
        for j, t in enumerate(tasks):
            ax = axes[i, j]
            for m, st in zip(MODES, ("-", "-", "--")):
                ys = [np.mean(list(R[t][f"{tg} | {m} | {d}"].values())) if f"{tg} | {m} | {d}" in R[t] else np.nan for d in dims]
                ax.plot(dims, ys, marker="o", ls=st, label=m)
            band = "réservoir cisaillé en bande, k = 10"
            if band in ref.get(t, {}):
                ax.axhline(np.mean(list(ref[t][band].values())), color="k", ls=":", lw=0.8)
            ax.set_xscale("log")
            if i == 0:
                ax.set_title(t, fontsize=9)
            if j == 0:
                ax.set_ylabel(tg + "\nlog10 NMSE", fontsize=8)
            if i == len(TANGENTS) - 1:
                ax.set_xlabel("dimension d")
    axes[0, -1].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(core.out_path("figures", "protocole_lie.png"), dpi=110)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--steps", type=int, default=core.STEPS)
    p.add_argument("--dims", type=int, nargs="*", default=DIMS)
    p.add_argument("--n_a", type=int, default=N_A)
    p.add_argument("--tasks", nargs="*")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--quick", action="store_true", help="vérification rapide (16 neurones, quelques pas)")
    a = p.parse_args()
    if a.quick:
        a.n, a.steps, a.dims, a.n_a = 16, 12, [10, 20], 60
    if not a.report:
        run(a, core.out_path("data", f"protocole_lie_s{a.seed}.json"))
    else:
        report(a)
