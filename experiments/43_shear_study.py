"""
Partie 5 — le réservoir cisaillé en bande : courbe en k, budget égal, portes, données fraîches.

Le cisaillé en bande (`Decomp(form="iwasawa_N", band=k)`) apprend k·N coefficients triangulaires en plus des biais et des
entrées. Quatre mesures, chacune écrite dans des fichiers « shards » (un par graine, pour lancer plusieurs processus à la
fois sur un seul GPU) :

    curve    la courbe en k : k = 0 (biais et entrées appris), 1, 2, 3, 5, 10, 20, 50, 100 et plein, cinq tâches
             synthétiques, 500 neurones ; erreur d'entraînement ET de test, pour voir si le gain tient à un k réglé
             tâche par tâche (sur-apprentissage) ; avec en plus le cisaillé avec amplitudes (A et N).
    budget   le même budget de paramètres que le GRU : les nouveaux modèles sont ajoutés à results/data/budget_inputs.json
             (mêmes graines 0 à 2, mêmes réservoirs, GRU, LRU et accordé déjà calculés et repris tels quels).
    gated    le cisaillé à portes, à 200 neurones : quels blocs d'un GRU peut-on figer ou cisailler (économie de 33 %,
             66 %, 99 % des poids), et que faut-il entraîner (entrées, portes, récurrence). Référence : GRU, LRU et accordé
             à 200 neurones, déjà dans budget_inputs.json.
    fresh    des données jamais vues : séries synthétiques régénérées avec d'autres graines, et (si le téléchargement
             marche) un système réel qui n'a servi à rien, EMPS. Les modèles et leurs réglages sont fixés dans ce
             fichier avant la mesure ; chaque modèle est mesuré une fois par graine.
    real     la courbe en k sur les systèmes réels (Silverbox, Wiener-Hammerstein, Cascaded Tanks).

    python experiments/43_shear_study.py curve --seeds 0 1 2
    python experiments/43_shear_study.py curve --report
    python experiments/43_shear_study.py gated --seeds 0 --quick          # vérification sur CPU
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

from hfrnn import core, tasks, task_map

SYN = ["Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)", "Lorenz (+10)", "Hénon (+3)"]
SIX = SYN + ["Wiener-Hammerstein"]
STD = SYN[:3]
BANDS = (1, 2, 3, 5, 10, 20, 50, 100)
ACC = "accordé"
SANSG = "k = 0 (biais et entrées appris, sans g)"


def band(k):
    return f"réservoir cisaillé en bande, k = {k}"


def band_g(k):
    return f"cisaillé en bande avec amplitudes (A et N), k = {k}"


# ---------------------------------------------------------------------------------------------------------------- modèles
CURVE = ([("réservoir classique", {"kind": "reservoir"}), (ACC, {"kind": "spectrum"}),
          (SANSG, {"kind": "spectrum", "learn_g": False})]
         + [(band(k), {"kind": "decomp", "form": "iwasawa_N", "band": k}) for k in BANDS]
         + [("réservoir cisaillé plein", {"kind": "decomp", "form": "iwasawa_N"})]
         + [(band_g(k), {"kind": "decomp", "form": "iwasawa_AN", "band": k}) for k in (3, 10, 30)]
         # contrôle : mêmes nombres de coefficients, positions tirées au hasard dans tout le triangle (la bande compte-t-elle ?)
         + [(f"cisaillé creux (positions au hasard), k = {k}", {"kind": "decomp", "form": "iwasawa_N", "band": k, "scatter": True})
            for k in (3, 10)]
         # contrôle : les mêmes nombres de coefficients appris directement dans W₀ (base des neurones, sans Q ni R)
         + [(f"contrôle : coefficients appris dans W₀ (au hasard), k = {k}",
             {"kind": "decomp", "form": "direct", "band": k}) for k in (3, 10)]
         + [(f"contrôle : coefficients appris dans le triangle de W₀, k = {k}",
             {"kind": "decomp", "form": "direct_tri", "band": k}) for k in (3, 10)])

# Budget égal : les modèles ajoutés à budget_inputs.json. N = 500 sauf mention.
BUDGET = ([(f"cisaillé en bande, N=500, k={k}", {"kind": "decomp", "form": "iwasawa_N", "band": k, "n": 500})
           for k in (1, 3, 10, 30, 100)]
          + [(f"cisaillé + g en bande, N=500, k={k}", {"kind": "decomp", "form": "iwasawa_AN", "band": k, "n": 500})
             for k in (3, 10, 30)]
          + [(f"cisaillé en bande, N={n}, k=10", {"kind": "decomp", "form": "iwasawa_N", "band": 10, "n": n})
             for n in (100, 200)])
BUDGET_REFS = ["GRU, 20 unités", "GRU, 60 unités", "GRU, 100 unités", "GRU, 200 unités", "GRU, 300 unités",
               "GRU, 500 unités", "LRU, 20 unités", "LRU, 60 unités", "LRU, 100 unités", "LRU, 200 unités",
               "LRU, 300 unités", "LRU, 500 unités", "accordé, N=500, rang 0", "accordé, N=500, rang 8",
               "accordé, N=500, rang 32", "accordé, N=500, rang 128", "sans g, N=500, rang 8", "sans g, N=500, rang 32",
               "sans g, N=500, rang 128"]

# Cisaillé à portes, à 200 neurones. maps = (porte z, porte r, candidat) ; k = 10.
def G(cell="gru", maps=("shear", "shear", "shear"), train=("in", "gate", "rec"), **kw):
    return {"kind": "shear_gated", "cell": cell, "maps": maps, "band": 10, "train": train, **kw, "n": 200}


GATED = [
    # économie de poids par rapport au GRU : une matrice récurrente sur trois cisaillée (−33 %), deux (−66 %), trois (−99 %)
    ("portes z et r apprises, candidat cisaillé (−33 %)", G(maps=("full", "full", "shear"))),
    ("porte z apprise, porte r et candidat cisaillés (−66 %)", G(maps=("full", "shear", "shear"))),
    ("portes cisaillées, candidat appris entier (−66 %)", G(maps=("shear", "shear", "full"))),
    ("portes et candidat cisaillés (−99 %)", G()),
    ("portes et candidat cisaillés, avec amplitudes g", G(learn_g=True)),
    ("portes neurone par neurone, candidat cisaillé (−99 %)", G(maps=("diag", "diag", "shear"))),
    ("porte d'entrée (type minGRU), candidat cisaillé", G(cell="mingru")),
    ("récurrence linéaire à porte d'entrée, candidat cisaillé", G(cell="lin_gate")),
    ("récurrence linéaire cisaillée, sans porte", G(cell="linear")),
    # que faut-il entraîner ? sur « portes et candidat cisaillés »
    ("entraîné sur les entrées seulement", G(train=("in",))),
    ("entraîné sur les portes seulement", G(train=("gate",))),
    ("entraîné sur la récurrence seulement", G(train=("rec",))),
    ("entraîné sur les entrées et la récurrence", G(train=("in", "rec"))),
    ("entraîné sur les entrées et les portes", G(train=("in", "gate"))),
    # portes apprises, candidat figé (Gated ESN) : le candidat n'apprend rien
    ("portes z et r apprises, candidat figé (−33 %)", G(maps=("full", "full", "frozen"), train=("in", "gate"))),
]
GATED_REFS = ["GRU, 200 unités", "LRU, 200 unités", "accordé : g + biais + entrées W_in",
              "contrôle : biais + entrées W_in, sans g", "réservoir classique"]

# Données fraîches : modèles et réglages fixés ici, avant la mesure.
FRESH = ([("réservoir classique", {"kind": "reservoir"}), (ACC, {"kind": "spectrum"}),
          (SANSG, {"kind": "spectrum", "learn_g": False})]
         + [(band(k), {"kind": "decomp", "form": "iwasawa_N", "band": k}) for k in (3, 10, 30)]
         + [(band_g(k), {"kind": "decomp", "form": "iwasawa_AN", "band": k}) for k in (3, 10)]
         + [("cisaillé à porte d'entrée (minGRU), k = 10", dict(G(cell="mingru"), n=500)),
            ("cisaillé linéaire à porte d'entrée, k = 10", dict(G(cell="lin_gate"), n=500)),
            ("GRU, 100 unités", {"kind": "gru", "rec_init": "ortho", "n": 100}),
            ("GRU, 500 unités", {"kind": "gru", "rec_init": "ortho", "n": 500}),
            ("LRU, 200 unités", {"kind": "lru", "n": 200})])
FRESH_TASKS = SIX

# Systèmes réels : courbe en k (en plus de ceux de nonlinear_benchmarks.json : k = 3, sans g, plein, accordé…).
REAL = ([(band(k), {"kind": "decomp", "form": "iwasawa_N", "band": k}) for k in (1, 10, 30, 100)]
        + [(band_g(k), {"kind": "decomp", "form": "iwasawa_AN", "band": k}) for k in (3, 10)]
        + [("cisaillé à porte d'entrée (minGRU), k = 10", dict(G(cell="mingru"), n=500)),
           ("cisaillé linéaire à porte d'entrée, k = 10", dict(G(cell="lin_gate"), n=500))])
REAL_REFS = ["réservoir classique", "réservoir accordé", "réservoir à biais et entrées appris (sans g)",
             band(3), "réservoir cisaillé", "GRU, init soignée", "LRU"]
REAL_REF_ALIAS = {"réservoir à biais et entrées appris (sans g)": SANSG}

MODES = {"curve": (CURVE, SYN, 500), "budget": (BUDGET, SIX, 500), "gated": (GATED, SYN, 200),
         "fresh": (FRESH, FRESH_TASKS, 500)}


# ---------------------------------------------------------------------------------------------------------------- données
def data(name):
    if name in STD:
        u, y = tasks.load(name)
        return u, y, core.n_train(u)
    return task_map.load(name)


def fresh_data(name):
    """Mêmes tâches, autres séries : graines de génération différentes (ou segment de la trajectoire jamais utilisé)."""
    if name == "Mackey-Glass (+15)":
        u, y = tasks.task_mackey_glass(8000)
        return task_map._norm(u[4000:], y[4000:], 3000)
    if name == "NARMA10":
        u, y = tasks.task_narma10(4000, seed=9001)
        return task_map._norm(u, y, 3000)
    if name == "Mémoire (délai 15)":
        u, y = tasks.task_delay(4000, 15, seed=9002)
        return task_map._norm(u, y, 3000)
    if name == "Lorenz (+10)":
        return task_map.lorenz(seed=9003)
    if name == "Hénon (+3)":
        return task_map.henon(seed=5)
    return task_map.wiener_hammerstein(seed=9005)


def load_module(fname, alias):
    spec = importlib.util.spec_from_file_location(alias, Path(__file__).with_name(fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------------------------- entraînement
def train_nmse(model, ut, yt, nt):
    """Erreur d'entraînement (lecture ridge ajustée sur les états d'entraînement et mesurée sur ces mêmes pas)."""
    import torch
    from hfrnn.torch_models import ridge, as_batch
    with torch.no_grad():
        X = model(as_batch(ut))[0]
        w = ridge(X[core.WASHOUT:nt], yt[core.WASHOUT:nt])
        A = torch.cat([X.double(), torch.ones(len(X), 1, dtype=torch.float64, device=X.device)], 1)
        e = (A[core.WASHOUT:nt] @ w - yt[core.WASHOUT:nt].double())
        return float((e ** 2).mean() / yt[core.WASHOUT:nt].double().var(unbiased=False))


def fit(u, y, nt, kw, seed, n, steps, dev):
    import torch
    from hfrnn import torch_models as TM
    kw = dict(kw)
    n = kw.pop("n", n)
    d = 1 if u.ndim == 1 else u.shape[1]
    Win, W = TM.make_reservoir(n, n_in=d, rho=core.RHO, in_scale=core.IN_SCALE / np.sqrt(d),
                               sparsity=core.SPARSITY, seed=seed)
    try:
        sc, info = TM.train_regression(u, y, Win, W, seed, steps=steps, device=dev, n_train=nt, return_model=True, **kw)
    except (RuntimeError, ValueError) as e:                 # entraînement qui diverge : noté, pas d'arrêt du processus
        print(f"  divergence ({type(e).__name__}: {str(e)[:80]})", flush=True)
        kind = kw.pop("kind", "convex")
        m = TM.build_model(Win, W, kind, seed, torch.float32, "cpu", **kw)
        return {"nmse": float("inf"), "train": float("inf"), "n_params": TM.n_trainable(m), "diverge": True}
    ut = torch.as_tensor(u, dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    tr = train_nmse(info["model"], ut, yt, nt)
    out = {"nmse": sc, "train": tr, "n_params": info["n_params"]}
    del info["model"]
    if dev == "cuda":
        torch.cuda.empty_cache()
    return out


def shard_path(mode, seed, quick=False):
    return core.out_path("data", f"shear_{mode}{'_quick' if quick else ''}_s{seed}.json")


def run_synthetic(a):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    models, task_list, n = MODES[a.mode]
    if a.quick:
        n = 24
    t0 = time.time()
    for seed in a.seeds:
        path = shard_path(a.mode, seed, a.quick)
        res = json.load(open(path)) if path.exists() else {"args": {"mode": a.mode, "n": n, "steps": a.steps}, "results": {}}
        for name in (a.tasks or task_list):
            u, y, nt = fresh_data(name) if a.mode == "fresh" else data(name)
            for label, kw in models:
                if a.only and not any(w in label for w in a.only):
                    continue
                cell = res["results"].setdefault(name, {}).setdefault(label, {})
                if str(seed) in cell:
                    continue
                kw2 = dict(kw)
                if a.quick and "n" in kw2:
                    kw2["n"] = min(kw2["n"], 24)
                cell[str(seed)] = fit(u, y, nt, kw2, seed, n, a.steps, dev)
                print(f"[{(time.time() - t0) / 60:6.1f} min] {name:20s} seed {seed} {label:58s} "
                      f"test {cell[str(seed)]['nmse']:.3g}  train {cell[str(seed)]['train']:.3g}", flush=True)
                json.dump(res, open(path, "w"), indent=1, ensure_ascii=False)


# ---------------------------------------------------------------------------------------------------------------- systèmes réels
def real_data(fresh, synthetic):
    NLB = load_module("30_nonlinear_benchmarks.py", "nlb30")
    if synthetic:
        return NLB, NLB.load_benchmarks(synthetic=True, quick=True)
    if not fresh:
        return NLB, NLB.load_benchmarks()
    import nonlinear_benchmarks as nb
    try:
        train, test = nb.EMPS()
        arr = lambda d: (np.asarray(d.u, float).ravel(), np.asarray(d.y, float).ravel())   # noqa: E731
        return NLB, {"EMPS": {"train": arr(train), "unit": "m", "scale": 1.0,
                              "tests": {"test (impulsions)": (*arr(test), int(test.state_initialization_window_length or 20))}}}
    except Exception as e:
        print("EMPS : téléchargement impossible", type(e).__name__, e, flush=True)
        return NLB, {}


def run_real(a):
    import torch
    from hfrnn import torch_models as TM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    NLB, bench = real_data(a.mode == "freshreal", a.synthetic)
    models = FRESH if a.mode == "freshreal" else REAL
    t0 = time.time()
    for seed in a.seeds:
        path = shard_path(a.mode, seed, a.quick or a.synthetic)
        res = json.load(open(path)) if path.exists() else {"args": {"mode": a.mode, "steps": a.steps}, "results": {}}
        for name, B in bench.items():
            if a.tasks and not any(t in name for t in a.tasks):
                continue
            u, y = B["train"]
            mu, su, my, sy = u.mean(), u.std(), y.mean(), y.std()
            un, yn = (u - mu) / su, (y - my) / sy
            tests = {k: ((tu - mu) / su, (ty - my) / sy, init) for k, (tu, ty, init) in B["tests"].items()}
            first = next(iter(tests.values()))
            uu = np.concatenate([un, first[0][:max(200, len(first[0]) // 10)]])
            yy = np.concatenate([yn, first[1][:len(uu) - len(un)]])
            ystd = sy * B["scale"]
            R = res["results"].setdefault(name, {})
            R["infos"] = {"unité": B["unit"], "points d'entraînement": len(un)}
            for label, kw in models:
                if a.only and not any(w in label for w in a.only):
                    continue
                cell = R.setdefault(label, {})
                if str(seed) in cell:
                    continue
                kw = dict(kw)
                n = kw.pop("n", 500)
                if a.quick or a.synthetic:
                    n = min(n, 24)
                Win, W = TM.make_reservoir(n, n_in=1, rho=core.RHO, in_scale=core.IN_SCALE, sparsity=core.SPARSITY,
                                           seed=seed)
                t1 = time.time()
                try:
                    _, info = TM.train_regression(uu, yy, Win, W, seed, steps=a.steps, device=dev, n_train=len(un),
                                                  return_model=True, **kw)
                    m = info["model"]
                    ut = torch.as_tensor(un, dtype=torch.float32, device=dev)
                    yt = torch.as_tensor(yn, dtype=torch.float32, device=dev)
                    cell[str(seed)] = {"tests": NLB.evaluate(m, ut, yt, tests, ystd, dev), "n_params": info["n_params"],
                                       "secondes": round(time.time() - t1, 1)}
                    del m, info
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    R.setdefault("erreurs", {})[f"{label}, seed {seed}"] = f"{type(e).__name__}: {e}"[:300]
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                    continue
                if dev == "cuda":
                    torch.cuda.empty_cache()
                main = next(iter(cell[str(seed)]["tests"].values()))
                print(f"[{(time.time() - t0) / 60:6.1f} min] {name:26s} seed {seed} {label:52s} "
                      f"RMSE {main['rmse']:.4g} {B['unit']}  NMSE {main['nmse']:.3g}  ({cell[str(seed)]['secondes']} s)",
                      flush=True)
                json.dump(res, open(path, "w"), indent=1, ensure_ascii=False)
        json.dump(res, open(path, "w"), indent=1, ensure_ascii=False)


# ---------------------------------------------------------------------------------------------------------------- rapport
def merge(a, b):
    for k, v in b.items():
        if k not in a:
            a[k] = v
        elif isinstance(a[k], dict) and isinstance(v, dict):
            merge(a[k], v)
    return a


def load_results(mode, quick=False):
    """Tous les shards du mode, fusionnés ; pour budget et gated, les résultats de budget_inputs.json en plus."""
    R = {}
    for f in sorted(glob.glob(str(core.out_path("data", f"shear_{mode}{'_quick' if quick else ''}_s[0-9]*.json")))):
        merge(R, json.load(open(f))["results"])
    if mode in ("budget", "gated") and not quick:
        base = core.out_path("data", "budget_inputs.json")
        if base.exists():
            old = json.load(open(base))["results"]
            keep = (BUDGET_REFS + [m[0] for m in BUDGET]) if mode == "budget" else GATED_REFS
            for t, cells in old.items():
                for lab, v in cells.items():
                    if lab in keep:
                        R.setdefault(t, {}).setdefault(lab, {})
                        for s, r in v.items():
                            R[t][lab].setdefault(s, r)
    return R


def scores(R, task, labels=None):
    cells = R.get(task, {})
    labels = labels or list(cells)
    seeds = sorted(set.intersection(*[set(cells[l]) for l in labels if l in cells and cells[l]]), key=int) if cells else []
    return {l: np.array([cells[l][s]["nmse"] for s in seeds]) for l in labels if l in cells and all(s in cells[l] for s in seeds)}, seeds


def gap(v, ref):
    d = np.log10(v / ref)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    thr = max(2 * se, 0.05) if np.isfinite(se) else 0.05
    verdict = "mieux" if d.mean() < -thr else "moins bien" if d.mean() > thr else "égal"
    return float(d.mean()), float(se), verdict


def report_synthetic(a):
    R = load_results(a.mode, a.quick)
    ref = {"curve": ACC, "budget": "GRU, 200 unités", "gated": "GRU, 200 unités", "fresh": ACC}[a.mode]
    out = {}
    for task in R:
        sc, seeds = scores(R, task)
        if not seeds or ref not in sc:
            continue
        print(f"\n{task} — {len(seeds)} graines (écart log10 apparié à « {ref} » ; négatif = mieux)")
        out[task] = {}
        for l, v in sc.items():
            m, se, verdict = gap(v, sc[ref])
            npar = R[task][l][seeds[0]]["n_params"]
            tr = np.array([R[task][l][s].get("train", np.nan) for s in seeds])
            out[task][l] = {"nmse": v.tolist(), "écart": m, "se": se, "verdict": verdict, "paramètres": npar,
                            "train": tr.tolist()}
            print(f"  {l:60s} {npar:8d} par.  test {np.exp(np.log(v).mean()):9.3g}  {m:+.2f} ± {se:.2f}  {verdict}")
    json.dump(out, open(core.out_path("data", f"shear_{a.mode}_summary.json"), "w"), indent=1, ensure_ascii=False)
    if a.mode == "curve":
        curve_extra(R, out)
        figure_curve(R)
    elif a.mode == "budget":
        figure_budget(R)
    elif a.mode == "gated":
        figure_gated(R, out)
    elif a.mode == "fresh":
        figure_fresh(R, out)


def curve_extra(R, out):
    """k choisi tâche par tâche (oracle) contre un seul k pour toutes les tâches : le gain tient-il à un k réglé ?"""
    ks = list(BANDS)
    tasks_ = [t for t in R if all(band(k) in R[t] for k in ks)]
    if not tasks_:
        return
    L = {}
    for t in tasks_:
        sc, seeds = scores(R, t, [band(k) for k in ks] + [ACC])
        L[t] = {k: np.log10(sc[band(k)]).mean() for k in ks}
        L[t]["acc"] = np.log10(sc[ACC]).mean()
    print("\nUn seul k pour toutes les tâches, ou le meilleur k de chaque tâche (écart log10 à l'accordé, moyenne des tâches)")
    summary = {}
    for k in ks:
        summary[k] = float(np.mean([L[t][k] - L[t]["acc"] for t in tasks_]))
        wins = sum(L[t][k] < L[t]["acc"] - 0.05 for t in tasks_)
        print(f"  k = {k:3d} partout : {summary[k]:+.2f}   bat l'accordé sur {wins} tâches sur {len(tasks_)}")
    oracle = float(np.mean([min(L[t][k] for k in ks) - L[t]["acc"] for t in tasks_]))
    print(f"  meilleur k de chaque tâche (choisi sur le test, borne optimiste) : {oracle:+.2f}")
    bestk = min(ks, key=lambda k: summary[k])
    print(f"  meilleur k unique : {bestk} ({summary[bestk]:+.2f}) ; coût de n'avoir qu'un k : {summary[bestk] - oracle:+.2f}")
    out["_un_seul_k"] = {"par k": summary, "oracle": oracle, "meilleur k unique": bestk}
    json.dump(out, open(core.out_path("data", "shear_curve_summary.json"), "w"), indent=1, ensure_ascii=False)


def figure_curve(R):
    ks = [0] + list(BANDS)
    xs = ks + [350]
    lab = lambda k: SANSG if k == 0 else band(k) if k != 350 else "réservoir cisaillé plein"   # noqa: E731
    present = [t for t in SYN if t in R]
    if not present:
        return
    fig, axes = plt.subplots(1, len(present), figsize=(3.6 * len(present), 3.6), squeeze=False)
    for ax, t in zip(axes[0], present):
        te, tr, xx = [], [], []
        for k, x in zip(xs, xs):
            c = R[t].get(lab(k), {})
            if c:
                te.append(np.exp(np.mean([np.log(v["nmse"]) for v in c.values()])))
                tr.append(np.exp(np.mean([np.log(v["train"]) for v in c.values()])))
                xx.append(max(x, 0.5))
        ax.plot(xx, te, "o-", color="tab:blue", label="test")
        ax.plot(xx, tr, "s--", color="tab:gray", label="entraînement")
        sc_ = [(k, R[t].get(f"cisaillé creux (positions au hasard), k = {k}", {})) for k in (3, 10)]
        sc_ = [(k, np.exp(np.mean([np.log(v["nmse"]) for v in c.values()]))) for k, c in sc_ if c]
        if sc_:
            ax.plot(*zip(*sc_), "x", color="tab:green", ms=8, label="positions au hasard")
        for lab_, mk, nm in (("contrôle : coefficients appris dans W₀ (au hasard), k = {}", "+", "dans W₀, au hasard"),
                             ("contrôle : coefficients appris dans le triangle de W₀, k = {}", "^", "dans le triangle de W₀")):
            pts = [(k, R[t].get(lab_.format(k), {})) for k in (3, 10)]
            pts = [(k, np.exp(np.mean([np.log(v["nmse"]) for v in c.values()]))) for k, c in pts if c]
            if pts:
                ax.plot(*zip(*pts), mk, color="tab:purple", ms=8, mfc="none", label=nm)
        c = R[t].get(ACC, {})
        if c:
            ax.axhline(np.exp(np.mean([np.log(v["nmse"]) for v in c.values()])), color="tab:red", ls=":", label="accordé (test)")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks([0.5, 1, 3, 10, 30, 100, 350]); ax.set_xticklabels(["0", "1", "3", "10", "30", "100", "plein"], fontsize=8)
        ax.set_title(t, fontsize=9); ax.set_xlabel("k (sur-diagonales apprises)")
    axes[0, 0].set_ylabel("NMSE (log)"); axes[0, 0].legend(fontsize=7)
    fig.suptitle("Cisaillé en bande à 500 neurones : erreur en fonction de k", fontsize=11)
    fig.tight_layout(); fig.savefig(core.out_path("figures", "shear_curve.png"), dpi=110)


def figure_budget(R):
    present = [t for t in ("Lorenz (+10)", "Hénon (+3)", "Wiener-Hammerstein", "NARMA10") if t in R]
    if not present:
        return
    fam = {"GRU": ("tab:orange", "o", lambda l: l.startswith("GRU")), "LRU": ("tab:pink", "o", lambda l: l.startswith("LRU")),
           "accordé de rang r": ("tab:blue", "s", lambda l: l.startswith("accordé, N=500")),
           "sans g, rang r": ("tab:cyan", "s", lambda l: l.startswith("sans g, N=500")),
           "cisaillé en bande": ("tab:green", "^", lambda l: l.startswith("cisaillé en bande")),
           "cisaillé en bande + g": ("tab:olive", "v", lambda l: l.startswith("cisaillé + g"))}
    fig, axes = plt.subplots(1, len(present), figsize=(4.2 * len(present), 3.8), squeeze=False)
    for ax, t in zip(axes[0], present):
        for name, (col, mk, test) in fam.items():
            pts = []
            for l, c in R[t].items():
                if test(l) and c:
                    pts.append((c["0"]["n_params"] if "0" in c else list(c.values())[0]["n_params"],
                                np.exp(np.mean([np.log(v["nmse"]) for v in c.values()]))))
            pts.sort()
            if pts:
                ax.plot(*zip(*pts), marker=mk, color=col, label=name, lw=1.2)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(t, fontsize=9); ax.set_xlabel("paramètres appris")
    axes[0, 0].set_ylabel("NMSE de test (log)"); axes[0, 0].legend(fontsize=7)
    fig.suptitle("À budget égal de paramètres : cisaillé en bande, accordé de rang r, GRU et LRU", fontsize=11)
    fig.tight_layout(); fig.savefig(core.out_path("figures", "shear_budget.png"), dpi=110)


def figure_gated(R, out):
    present = [t for t in SYN if t in out]
    if not present:
        return
    labels = [m[0] for m in GATED] + ["GRU, 200 unités", "LRU, 200 unités", "accordé : g + biais + entrées W_in"]
    fig, axes = plt.subplots(1, len(present), figsize=(3.9 * len(present), 5.8), squeeze=False, sharey=True)
    for ax, t in zip(axes[0], present):
        ys = [l for l in labels if l in out[t]]
        v = [np.exp(np.mean(np.log(out[t][l]["nmse"]))) for l in ys]
        col = ["tab:orange" if l.startswith("GRU") else "tab:pink" if l.startswith("LRU") else "tab:blue"
               if l.startswith("accordé") else "tab:green" for l in ys]
        ax.hlines(range(len(ys)), min(v) / 3, v, color=col, lw=1)
        ax.scatter(v, range(len(ys)), c=col, s=22)
        ax.set_xscale("log"); ax.set_title(t, fontsize=9); ax.invert_yaxis(); ax.set_xlabel("NMSE de test (log)")
        ax.set_yticks(range(len(ys))); ax.set_yticklabels([f"{l} [{out[t][l]['paramètres'] // 1000} k]" if out[t][l]["paramètres"] >= 1000
                                                          else l for l in ys], fontsize=7)
    fig.suptitle("Cisaillé à portes, 200 neurones : économiser des poids du GRU", fontsize=11)
    fig.tight_layout(); fig.savefig(core.out_path("figures", "shear_gated.png"), dpi=110)


def figure_fresh(R, out):
    present = [t for t in FRESH_TASKS if t in out]
    if not present:
        return
    labels = [m[0] for m in FRESH]
    fig, axes = plt.subplots(1, len(present), figsize=(3.4 * len(present), 5.2), squeeze=False, sharey=True)
    for ax, t in zip(axes[0], present):
        ys = [l for l in labels if l in out[t]]
        v = [np.exp(np.mean(np.log(out[t][l]["nmse"]))) for l in ys]
        col = ["tab:orange" if l.startswith("GRU") else "tab:pink" if l.startswith("LRU") else "tab:gray"
               if l == "réservoir classique" else "tab:blue" if l in (ACC, SANSG) else "tab:green" for l in ys]
        ax.hlines(range(len(ys)), min(v) / 3, v, color=col, lw=1)
        ax.scatter(v, range(len(ys)), c=col, s=22)
        ax.set_xscale("log"); ax.set_title(t, fontsize=9); ax.invert_yaxis(); ax.set_xlabel("NMSE de test (log)")
        ax.set_yticks(range(len(ys))); ax.set_yticklabels(ys, fontsize=7)
    fig.suptitle("Données fraîches : séries régénérées, réglages fixés avant la mesure", fontsize=11)
    fig.tight_layout(); fig.savefig(core.out_path("figures", "shear_fresh.png"), dpi=110)


def report_real(a):
    R = {}
    for f in sorted(glob.glob(str(core.out_path("data", f"shear_{a.mode}{'_quick' if a.quick or a.synthetic else ''}_s[0-9]*.json")))):
        merge(R, json.load(open(f))["results"])
    if a.mode == "real":                      # les modèles déjà mesurés dans nonlinear_benchmarks.json
        p = core.out_path("data", "nonlinear_benchmarks.json")
        if p.exists():
            old = json.load(open(p))["results"]
            for b, cells in old.items():
                if "erreur" in cells:
                    continue
                for lab in REAL_REFS:
                    if lab in cells:
                        R.setdefault(b, {}).setdefault(REAL_REF_ALIAS.get(lab, lab), {})
                        for s, r in cells[lab].items():
                            R[b][REAL_REF_ALIAS.get(lab, lab)].setdefault(s, r)
    out = {}
    for b, cells in R.items():
        if "erreur" in cells:
            continue
        unit = cells.get("infos", {}).get("unité", "")
        rows = {}
        for lab, c in cells.items():
            if lab in ("infos", "erreurs") or not c:
                continue
            for test in next(iter(c.values()))["tests"]:
                v = np.array([x["tests"][test]["rmse"] for x in c.values()])
                nm = np.array([x["tests"][test]["nmse"] for x in c.values()])
                rows.setdefault(test, {})[lab] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0,
                                                  float(np.exp(np.log(nm).mean())), next(iter(c.values()))["n_params"], len(v))
        for test, rr in rows.items():
            print(f"\n{b} — {test} ({unit})")
            for lab, (m, se, nm, npar, ns) in sorted(rr.items(), key=lambda kv: kv[1][0]):
                print(f"  {lab:56s} {npar:8d} par.  RMSE {m:9.4g} ± {se:.2g}  NMSE {nm:.3g}  ({ns} graines)")
            out[f"{b} — {test}"] = {l: {"rmse": v[0], "se": v[1], "nmse": v[2], "paramètres": v[3], "graines": v[4]}
                                    for l, v in rr.items()}
    json.dump(out, open(core.out_path("data", f"shear_{a.mode}_summary.json"), "w"), indent=1, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["curve", "budget", "gated", "fresh", "real", "freshreal"])
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=core.STEPS)
    ap.add_argument("--tasks", nargs="*")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="real/freshreal : séries simulées au lieu des téléchargements")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.steps = 10
    if a.mode in ("real", "freshreal"):
        if not a.report:
            run_real(a)
        report_real(a)
    else:
        if not a.report:
            run_synthetic(a)
        report_synthetic(a)


if __name__ == "__main__":
    main()
