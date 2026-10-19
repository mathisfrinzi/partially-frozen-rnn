"""Partie 7 — les quatre formes de couplage réservoir/RNN contre les références.

    python experiments/35_coupled_forms.py --n 100 --seeds 3
    python experiments/35_coupled_forms.py --report
"""
import argparse, json, time
import numpy as np
from hfrnn import core, tasks, task_map
from hfrnn import torch_models as TM

LR = core.LR
MODELS = [
    ("réservoir classique", {"kind": "reservoir"}),
    ("RNN", {"kind": "convex", "alpha": 1.0}),
    ("RNN, lr × 0.1", {"kind": "convex", "alpha": 1.0, "lr": 0.1 * LR}),
    ("réservoir accordé", {"kind": "spectrum"}),
    ("GRU, init soignée", {"kind": "gru", "rec_init": "ortho"}),
    ("forme 1 : duo croisé, lu côté RNN", {"kind": "coupled", "form": "duo1"}),
    ("forme 2 : duo croisé, lu côté réservoir", {"kind": "coupled", "form": "duo2"}),
    ("forme 3 : relais aléatoire", {"kind": "coupled", "form": "relay_rand"}),
    ("forme 4 : relais appris", {"kind": "coupled", "form": "relay_learn"}),
    # contrôles ajoutés après avoir vu les formes 3 et 4 gagner (Hénon) : la nouveauté vient-elle du figé ou de la profondeur ?
    ("contrôle : réservoir classique, readout carré", {"kind": "reservoir", "square": True}),
    ("contrôle : réservoir accordé, readout carré", {"kind": "spectrum", "square": True}),
    ("contrôle : RNN, readout carré", {"kind": "convex", "alpha": 1.0, "square": True}),
    ("contrôle : relais tout appris (type DT-RNN)", {"kind": "coupled", "form": "relay_rand", "learn_R": True}),
    # à 500 neurones, les formes à relais héritent du learning rate par défaut, déjà trop grand pour le RNN à cette taille :
    # mêmes modèles avec lr × 0.1.
    ("forme 3 : relais aléatoire, lr × 0.1", {"kind": "coupled", "form": "relay_rand", "lr": 0.1 * LR}),
    ("forme 4 : relais appris, lr × 0.1", {"kind": "coupled", "form": "relay_learn", "lr": 0.1 * LR}),
    ("contrôle : relais tout appris, lr × 0.1",
     {"kind": "coupled", "form": "relay_rand", "learn_R": True, "lr": 0.1 * LR}),
]
STD = ["Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)"]
MAP = ["Lorenz (+10)", "Hénon (+3)"]
REF = "réservoir accordé"


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
            Win, W = TM.make_reservoir(a.n, n_in=d, rho=core.RHO, in_scale=core.IN_SCALE / np.sqrt(d),
                                       sparsity=core.SPARSITY, seed=seed)
            for label, kw in MODELS:
                if a.only and not any(w in label for w in a.only): continue
                cell = res["results"].setdefault(name, {}).setdefault(label, {})
                if str(seed) in cell: continue
                sc, info = TM.train_regression(u, y, Win, W, seed, steps=a.steps, device=dev, n_train=nt, **dict(kw))
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
            d = np.log10(v / sc[REF]) if REF in sc else np.full(seeds, np.nan); se = d.std(ddof=1) / np.sqrt(seeds) if seeds > 1 else float("nan")
            verdict = "mieux" if d.mean() < -max(2 * se, .05) else "moins bien" if d.mean() > max(2 * se, .05) else "égal"
            print(f"  {l:42s} {R[l]['0']['n_params']:7d} par.  NMSE {v.mean():9.3g}  vs accordé {d.mean():+.2f} ± {se:.2f}  {verdict}")
            out[name][l] = {"nmse": v.tolist(), "vs accordé": float(d.mean()), "se": float(se), "verdict": verdict,
                            "paramètres": R[l]["0"]["n_params"]}
    json.dump(out, open(path.replace(".json", "_summary.json"), "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100); p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--steps", type=int, default=core.STEPS); p.add_argument("--tasks", nargs="*")
    p.add_argument("--report", action="store_true"); p.add_argument("--tag", default=""); p.add_argument("--only", nargs="*")
    a = p.parse_args()
    path = str(core.out_path("data", f"coupled_forms_n{a.n}{a.tag}.json"))
    if not a.report: run(a, path)
    report(path)
