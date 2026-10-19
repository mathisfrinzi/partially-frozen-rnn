"""Grilles d'expériences en parallèle, avec cache sur disque et comparaisons appariées.

Une expérience élémentaire est un dictionnaire (« spec ») :
    {"kind": "convex", "alpha": 0.1, "task": "NARMA10", "seed": 3}

kind (nom dans les résultats) :
    "A"         réservoir classique    W = W₀, rien d'appris sauf le readout
    "convex"    mélange fixe           W = (1-α) W₀ + α ΔW       (alpha, lr_mult, lr_mult_in)
                                       α = 1 : RNN ; lr_mult = 0.1, lr_mult_in = 1 : RNN freiné
    "lalpha"    mélange réglé          α appris                   (alpha0)
    "lowrank"   réservoir corrigé      W = W₀ + U Vᵀ              (rank)
    "sparse"    réservoir retouché     W = W₀ + M ⊙ ΔW
    "supermask" réservoir sculpté      W = m ⊙ W₀, m appris       (mode, keep, win, mult, s0, rescale)
                (élagué si m binaire, repondéré si gain libre)
    "rotation"  réservoir tourné       W = Q W₀ Qᵀ, Q orthogonale (win, mult)
    "spectrum"  réservoir accordé      W = U diag(σ ⊙ g) Vᵀ       (mult)
    "combo"     combinaisons (partie 4) W = m ⊙ [U diag(σ ⊙ g) Vᵀ ou W₀] + ΔW lent
                                       (spectrum, mask, slow, win, mask_mult, spec_mult)
options : n (neurones), rho (rayon spectral), sparsity (densité du réservoir), lr (lr global),
init (readout de départ).

Chaque résultat est mis en cache dans results/data/cache.json : le code est déterministe,
donc une même spec redonne toujours le même résultat et n'est jamais recalculée.
"""
import os
import json
import time
from multiprocessing import Pool

import numpy as np

from . import core, params as P, tasks

DEFAULTS = {"n": 100, "rho": 0.9, "scale": 1.0, "init": "zero"}


def key_of(spec):
    return json.dumps(dict(DEFAULTS, **spec), sort_keys=True)


def run_one(spec):
    s = dict(DEFAULTS, **spec)
    u, y = tasks.load(s["task"])
    Win_r, W_r = core.make_reservoir(np.random.default_rng(s["seed"]), n=s["n"], rho=s["rho"],
                                     sparsity=s.get("sparsity"))
    Win_r, W_r = s["scale"] * Win_r, s["scale"] * W_r
    k = s["kind"]
    if k == "A":
        return P.pure_reservoir(u, y, Win_r, W_r), {}
    param = {"convex": lambda: P.Convex(s.get("alpha", 1.0), s.get("lr_mult", 1.0), s.get("lr_mult_in")),
             "lalpha": lambda: P.LearnedAlpha(s.get("alpha0", 0.1)),
             "lowrank": lambda: P.LowRank(s["rank"]),
             "sparse": lambda: P.SparseMask(),
             "supermask": lambda: P.SuperMask(s.get("mode", "sigmoid"), s.get("keep", 0.5), s.get("win", "free"),
                                              s.get("mult", 10.0), s.get("s0", 3.0), s.get("rescale", False)),
             "rotation": lambda: P.Rotation(s.get("win", "free"), s.get("mult", 1.0)),
             "spectrum": lambda: P.SpectrumOnly(s.get("mult", 1.0)),
             "combo": lambda: P.Combo(s.get("spectrum", False), s.get("mask", False), s.get("slow"),
                                      s.get("win", "free"), s.get("mask_mult", 50.0), s.get("spec_mult", 1.0))}[k]()
    return P.train(param, u, y, Win_r, W_r, s["seed"], lr=s.get("lr"), init=s["init"])


def _job(spec):
    t0 = time.time()
    score, info = run_one(spec)
    return key_of(spec), {"nmse": score, "info": info, "sec": round(time.time() - t0, 1)}


def _cache_path():
    return core.out_path("data", "cache.json")


def load_cache():
    try:
        with open(_cache_path()) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def run_specs(specs, procs=None, verbose=True):
    """Calcule les specs absentes du cache ; renvoie {clé: résultat} pour toutes."""
    procs = procs or int(os.environ.get("HFRNN_PROCS", 2))
    cache = load_cache()
    todo = [s for s in specs if key_of(s) not in cache]
    todo.sort(key=lambda s: -dict(DEFAULTS, **s)["n"])          # les plus longs d'abord
    if verbose:
        print(f"{len(specs)} runs, {len(specs) - len(todo)} en cache, {len(todo)} à calculer", flush=True)
    t0 = time.time()
    if todo:
        with Pool(procs) as pool:
            for i, (k, r) in enumerate(pool.imap_unordered(_job, todo, chunksize=1)):
                cache[k] = r
                if i % 8 == 0 or i == len(todo) - 1:
                    if verbose:
                        print(f"  [{time.time() - t0:6.0f}s] {i + 1}/{len(todo)}", flush=True)
                    with open(_cache_path(), "w") as f:
                        json.dump(cache, f)
    return {key_of(s): cache[key_of(s)] for s in specs}


def grid(models, task_list, seeds):
    """Toutes les specs (modèle × tâche × seed)."""
    return [dict(m, task=t, seed=s) for m in models for t in task_list for s in seeds]


def scores(results, spec, seeds):
    return np.array([results[key_of(dict(spec, seed=s))]["nmse"] for s in seeds])


def infos(results, spec, seeds, field):
    return np.array([results[key_of(dict(spec, seed=s))]["info"].get(field, np.nan) for s in seeds])


def paired(a, ref):
    """Écart apparié log10(a / ref) : moyenne, erreur standard, t."""
    d = np.log10(np.asarray(a) / np.asarray(ref))
    se = d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), se, (d.mean() / se if se > 0 else 0.0)


def table(rows, ref_label, title):
    """rows : [(label, scores)]. Affiche et renvoie une table avec écart apparié à ref_label."""
    ref = dict(rows)[ref_label]
    n = len(ref)
    print(f"\n{title}  ({n} seeds ; écart = log10(NMSE / {ref_label}))")
    print(f"{'modèle':>30s} {'NMSE':>10s} {'± err.std':>10s} {'écart':>15s} {'t':>7s}")
    out = []
    for label, a in rows:
        m, se = a.mean(), a.std(ddof=1) / np.sqrt(n)
        d, dse, t = paired(a, ref)
        print(f"{label:>30s} {m:10.6f} {se:10.6f} {d:+8.3f} ±{dse:5.3f} {t:7.2f}")
        out.append({"label": label, "mean": float(m), "se": float(se), "d": float(d), "dse": float(dse),
                    "t": float(t), "values": [float(x) for x in a]})
    return out
