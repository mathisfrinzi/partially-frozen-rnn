"""Parties 4.2 et 4.9 — budget égal de paramètres (accordé de rang r contre GRU et LRU) et qui apprend quoi dans le réservoir accordé.

    python experiments/37_budget_and_inputs.py --seeds 3
    python experiments/37_budget_and_inputs.py --report
"""
import argparse, json, time
import numpy as np
from hfrnn import core, tasks, task_map
from hfrnn import torch_models as TM

LR = core.LR
# Expérience A (budget) : à budget de paramètres appris comparable, le réservoir accordé (rang r croissant) contre GRU et LRU
# (nombre d'unités ajusté).
BUDGET = [("accordé, N=500, rang %d" % r, {"kind": "tuned", "rank": r, "n": 500}) for r in (0, 8, 32, 128, 256)] + \
         [("GRU, %d unités" % n, {"kind": "gru", "rec_init": "ortho", "n": n}) for n in (20, 60, 100, 200, 300, 500)] + \
         [("LRU, %d unités" % n, {"kind": "lru", "n": n}) for n in (20, 60, 100, 200, 300, 500)]
# Expérience B (entrées) : qui apprend quoi dans le réservoir accordé ? g seul, + biais, + entrées W_in, tout.
INPUTS = [("accordé : amplitudes g seules", {"kind": "spectrum", "learn_win": False, "train_bias": False, "n": 200}),
          ("accordé : g + biais", {"kind": "spectrum", "learn_win": False, "n": 200}),
          ("accordé : g + entrées W_in", {"kind": "spectrum", "train_bias": False, "n": 200}),
          ("accordé : g + biais + entrées W_in", {"kind": "spectrum", "n": 200}),
          ("accordé intégral (masque sur W_in)", {"kind": "combo", "spectrum": True, "win": "mask", "n": 200}),
          ("réservoir classique", {"kind": "reservoir", "n": 200}),
          # contrôles : W_in et biais SANS les amplitudes g (récurrence strictement celle du réservoir).
          ("contrôle : biais + entrées W_in, sans g", {"kind": "spectrum", "learn_g": False, "n": 200}),
          ("contrôle : biais seul, sans g", {"kind": "spectrum", "learn_g": False, "learn_win": False, "n": 200}),
          ("contrôle : entrées W_in seules, sans g", {"kind": "spectrum", "learn_g": False, "train_bias": False, "n": 200})]
# « Sans g + rang r » : biais + W_in + correction U Vᵀ de rang r, récurrence = W₀ + UVᵀ (partie 4.9).
SANSG = [("sans g, N=500, rang %d" % r, {"kind": "tuned", "rank": r, "learn_g": False, "n": 500}) for r in (8, 32, 128)]
SANSG_TASKS = ["Lorenz (+10)", "Hénon (+3)", "Wiener-Hammerstein"]
MODELS = BUDGET + INPUTS + SANSG
STD = ["Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)"]
MAP = ["Lorenz (+10)", "Hénon (+3)"]
REF = "accordé, N=500, rang 0"
REF2 = "accordé : g + biais + entrées W_in"


def data(name):
    if name in STD:
        u, y = tasks.load(name); return u, y, core.n_train(u)
    return task_map.load(name)


def run(a, path):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try: res = json.load(open(path))
    except Exception: res = {"args": vars(a), "results": {}}
    t0 = time.time()
    for name in (a.tasks or STD + MAP):
        u, y, nt = data(name)
        d = 1 if u.ndim == 1 else u.shape[1]
        for seed in range(a.seeds):
            for label, kw in MODELS:
                kw = dict(kw); n = kw.pop("n")
                Win, W = TM.make_reservoir(n, n_in=d, rho=core.RHO, in_scale=core.IN_SCALE / np.sqrt(d),
                                           sparsity=core.SPARSITY, seed=seed)
                if a.only and not any(w in label for w in a.only): continue
                cell = res["results"].setdefault(name, {}).setdefault(label, {})
                if str(seed) in cell: continue
                sc, info = TM.train_regression(u, y, Win, W, seed, steps=a.steps, device=dev, n_train=nt, **kw)
                cell[str(seed)] = {"nmse": sc, "n_params": info["n_params"]}
                print(f"[{(time.time()-t0)/60:5.1f} min] {name:20s} seed {seed} {label:42s} {sc:.3g}", flush=True)
                json.dump(res, open(path, "w"), indent=1)


def report(path):
    res = json.load(open(path)); out = {}
    for name, R in res["results"].items():
        seeds = min(len(v) for v in R.values())
        if not seeds: continue
        sc = {l: np.array([R[l][str(s)]["nmse"] for s in range(seeds)]) for l in R if len(R[l]) >= seeds}
        print(f"\n{name} — {seeds} graines (écart log10 apparié ; négatif = mieux ; seuil −max(2·SE, 0.05))")
        out[name] = {}
        for l, v in sc.items():
            ref = REF2 if l in [m[0] for m in INPUTS] else REF
            d = np.log10(v / sc[ref]) if ref in sc else np.full(seeds, np.nan); se = d.std(ddof=1) / np.sqrt(seeds) if seeds > 1 else float("nan")
            verdict = "mieux" if d.mean() < -max(2 * se, .05) else "moins bien" if d.mean() > max(2 * se, .05) else "égal"
            print(f"  {l:42s} {R[l]['0']['n_params']:7d} par.  NMSE {v.mean():9.3g}  vs accordé {d.mean():+.2f} ± {se:.2f}  {verdict}")
            out[name][l] = {"nmse": v.tolist(), "vs accordé": float(d.mean()), "se": float(se), "verdict": verdict,
                            "paramètres": R[l]["0"]["n_params"]}
    json.dump(out, open(path.replace(".json", "_summary.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=core.STEPS); p.add_argument("--tasks", nargs="*")
    p.add_argument("--report", action="store_true"); p.add_argument("--tag", default=""); p.add_argument("--only", nargs="*")
    a = p.parse_args()
    path = str(core.out_path("data", f"budget_inputs{a.tag}.json"))
    if not a.report: run(a, path)
    report(path)
