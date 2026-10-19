"""
Partie 7.5 — Pourquoi le readout carré marche-t-il si bien ? Quatre contrôles (100 neurones par défaut, GPU conseillé).

Le readout carré lit [x, x²] (2 N caractéristiques) au lieu de x. En 3.3, il fait gagner un facteur 1,5 à 8 à l'accordé
selon les tâches. Trois explications possibles, que ces contrôles séparent :

    (a) il y a simplement plus de caractéristiques                  -> réservoir accordé à 2 N neurones, readout linéaire
    (b) la symétrie impaire du tanh est brisée (fonctions paires)   -> [x, x³] (impaire) contre [x, x²] et [x, |x|] (paires)
    (c) il faut des non-linéarités d'ordre 2 entre états différents -> [x, x_i·x_j] sur N paires tirées au hasard

Modèles (même nombre de caractéristiques, 2 N, sauf mention ; « réservoir classique + cube » ajouté après les premiers résultats) :
    accordé                       référence (N caractéristiques, 3 N paramètres appris)
    accordé + carré / cube / |x| / produits    2 N caractéristiques, mêmes paramètres appris
    accordé, 2 N neurones         2 N caractéristiques, 3 × 2 N paramètres appris (contrôle (a))
    réservoir classique, + carré, 2 N neurones   mêmes comparaisons sans aucun apprentissage récurrent

Contrôle de la parité propre : le réservoir classique (sans biais, états impairs) avec le cube.

    python experiments/41_readout_features.py --seeds 3        # N = 100, ~40 min sur un T4
    python experiments/41_readout_features.py --n 250 --seeds 3
    python experiments/41_readout_features.py --report
"""
import argparse
import json
import time

import numpy as np

from hfrnn import core, tasks, task_map
from hfrnn import torch_models as TM

STD = ["Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)"]
MAP = ["Lorenz (+10)", "Hénon (+3)"]
REF = "réservoir accordé"


def models(n):
    """(libellé, options, taille du réservoir)."""
    acc = {"kind": "spectrum"}
    return [
        ("réservoir classique", {"kind": "reservoir"}, n),
        ("réservoir classique + carré", {"kind": "reservoir", "square": True}, n),
        ("réservoir classique + cube", {"kind": "reservoir", "feat": "cube"}, n),
        (f"réservoir classique, {2 * n} neurones", {"kind": "reservoir"}, 2 * n),
        ("réservoir accordé", dict(acc), n),
        ("accordé + carré  [x, x²]", {**acc, "square": True}, n),
        ("accordé + cube  [x, x³] (impair)", {**acc, "feat": "cube"}, n),
        ("accordé + valeur absolue  [x, |x|]", {**acc, "feat": "abs"}, n),
        ("accordé + produits de paires  [x, xᵢxⱼ]", {**acc, "feat": "pairs"}, n),
        (f"accordé, {2 * n} neurones", dict(acc), 2 * n),
    ]


def data(name):
    if name in STD:
        u, y = tasks.load(name)
        return u, y, core.n_train(u)
    return task_map.load(name)


def run(a, path):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        res = json.load(open(path))
    except Exception:
        res = {"args": vars(a), "results": {}}
    t0 = time.time()
    for name in (a.tasks or STD + MAP):
        u, y, nt = data(name)
        d = 1 if u.ndim == 1 else u.shape[1]
        for seed in range(a.seeds):
            for label, kw, n in models(a.n):
                if a.only and not any(w in label for w in a.only):
                    continue
                cell = res["results"].setdefault(name, {}).setdefault(label, {})
                if str(seed) in cell:
                    continue
                Win, W = TM.make_reservoir(n, n_in=d, rho=core.RHO, in_scale=core.IN_SCALE / np.sqrt(d),
                                           sparsity=core.SPARSITY, seed=seed)
                sc, info = TM.train_regression(u, y, Win, W, seed, steps=a.steps, device=dev, n_train=nt, **dict(kw))
                cell[str(seed)] = {"nmse": sc, "n_params": info["n_params"]}
                print(f"[{(time.time() - t0) / 60:5.1f} min] {name:20s} seed {seed} {label:46s} {sc:.3g}", flush=True)
                json.dump(res, open(path, "w"), indent=1)


def report(path):
    res = json.load(open(path))
    out = {}
    for name, R in res["results"].items():
        seeds = min(len(v) for v in R.values())
        if not seeds or REF not in R:
            continue
        sc = {l: np.array([R[l][str(s)]["nmse"] for s in range(seeds)]) for l in R if len(R[l]) >= seeds}
        print(f"\n{name} — {seeds} graines (écart log10 apparié à l'accordé sans carré ; négatif = mieux ; "
              "seuil −max(2·SE, 0.05))")
        out[name] = {}
        for l, v in sc.items():
            d = np.log10(v / sc[REF])
            se = d.std(ddof=1) / np.sqrt(seeds) if seeds > 1 else float("nan")
            verdict = "mieux" if d.mean() < -max(2 * se, .05) else "moins bien" if d.mean() > max(2 * se, .05) else "égal"
            print(f"  {l:46s} {R[l]['0']['n_params']:6d} par.  NMSE {v.mean():9.3g}  vs accordé {d.mean():+.2f} ± {se:.2f}  {verdict}")
            out[name][l] = {"nmse": v.tolist(), "vs accordé": float(d.mean()), "se": float(se), "verdict": verdict,
                            "paramètres": R[l]["0"]["n_params"]}
    json.dump(out, open(path.replace(".json", "_summary.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=core.STEPS)
    p.add_argument("--tasks", nargs="*")
    p.add_argument("--only", nargs="*")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    path = str(core.out_path("data", f"readout_features_n{a.n}.json"))
    if not a.report:
        run(a, path)
    report(path)
