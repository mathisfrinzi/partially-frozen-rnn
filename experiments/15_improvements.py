"""
Partie 4.4 — Améliorer le réservoir accordé (W = U · diag(σ ⊙ g) · Vᵀ, seuls g, W_in, b et le readout appris).

Variantes testées (voir src/hfrnn/torch_tuned.py) :
    borné            g ∈ [0, 2]
    + fuite          vitesse propre par neurone : x ← (1 − a) x + a tanh(…)
    valeurs propres  module et phase de chaque mode propre au lieu des valeurs singulières
    paramétrique     g = fonction lisse de 8 coefficients (au lieu de N amplitudes)
    profond          deux réservoirs accordés de N/2 neurones empilés
    + rang 4         accordé + correction de rang 4 (les deux meilleurs modèles à 500 neurones)

Même protocole que le reste de l'étude (1 500 pas, readout ridge), version PyTorch (CPU ou GPU).

    python experiments/15_improvements.py                  # N = 200, 3 seeds (tri sur CPU, ~3 h sur 2 cœurs)
    python experiments/15_improvements.py --n 500          # confirmation sur GPU
    python experiments/15_improvements.py --report

Résultats sauvegardés après chaque entraînement ; une relance reprend là où le calcul s'était arrêté.
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
DEFAULT_N = 200
TASKS = ["Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)"]
RNNS = ["RNN", "RNN, lr global × 0.3", "RNN, lr global × 0.1"]
REF = "réservoir accordé"
REF_LABEL = "meilleur RNN"                   # nom de la référence dans les tableaux
PREFIX = "improvements"                      # nom des fichiers de résultats (réutilisé par 18_spectrum_controls.py)
TITLE = ("Partie 4.2 — variantes du réservoir accordé (cyan). Tirets rouges : meilleur RNN ; "
         "pointillés bleus : réservoir accordé.")
MODELS = [
    ("réservoir classique", {"kind": "reservoir"}, "gray"),
    ("RNN", {"kind": "convex", "alpha": 1.0}, "tab:red"),
    ("RNN, lr global × 0.3", {"kind": "convex", "alpha": 1.0, "lr": 0.3 * LR}, "salmon"),
    ("RNN, lr global × 0.1", {"kind": "convex", "alpha": 1.0, "lr": 0.1 * LR}, "salmon"),
    ("RNN freiné", {"kind": "convex", "alpha": 1.0, "lr_mult": 0.1, "lr_mult_in": 1.0}, "tab:brown"),
    ("réservoir corrigé de rang 4", {"kind": "lowrank", "rank": 4}, "tab:olive"),
    ("réservoir sculpté intégral, dense",
     {"kind": "supermask", "mode": "sigmoid", "win": "mask", "mult": 50.0, "sparsity": 1.0}, "tab:purple"),
    ("réservoir accordé", {"kind": "spectrum"}, "tab:blue"),
    # partie 4.2
    ("accordé borné", {"kind": "tuned", "gmax": 2.0}, "tab:cyan"),
    ("accordé + fuite", {"kind": "tuned", "leak": True}, "tab:cyan"),
    ("accordé, valeurs propres", {"kind": "tuned", "basis": "eig"}, "tab:cyan"),
    ("accordé paramétrique (8 coef.)", {"kind": "tuned", "poly": 8}, "tab:cyan"),
    ("accordé profond (2 × N/2)", {"kind": "deep_tuned"}, "tab:cyan"),
    ("accordé + rang 4", {"kind": "tuned", "rank": 4}, "tab:cyan"),
    # deuxième vague
    ("accordé sélectif", {"kind": "tuned", "selective": True}, "tab:cyan"),
    ("accordé + gains par neurone", {"kind": "tuned", "neuron_gains": True}, "tab:cyan"),
    ("accordé, pénalité 1e-4", {"kind": "tuned", "penalty": 1e-4}, "tab:cyan"),
    ("accordé, pénalité 1e-3", {"kind": "tuned", "penalty": 1e-3}, "tab:cyan"),
    ("accordé + 32 rotations", {"kind": "tuned", "rotations": 32}, "tab:cyan"),
    # contrôle : sensibilité au learning rate des amplitudes
    ("contrôle : accordé, lr de g × 0.3", {"kind": "spectrum", "mult": 0.3}, "lightsteelblue"),
    ("contrôle : accordé, lr de g × 3", {"kind": "spectrum", "mult": 3.0}, "lightsteelblue"),
]
NEW = [l for l, m, _ in MODELS if m["kind"] in ("tuned", "deep_tuned")]
VARIANTS_DOC = """
Deuxième vague :
    sélectif         les amplitudes dépendent de l'entrée : g(t) = g + G·u(t)
    gains par neurone   W = diag(d_L)·U·diag(σ ⊙ g)·Vᵀ·diag(d_R)
    pénalité         rester près du réservoir : λ · moyenne((g − 1)²) ajoutée à la perte
    32 rotations     quelques rotations apprises entre paires de directions
    contrôles        learning rate des amplitudes × 0.3 et × 3
"""


def run(a, path, res):
    import torch
    from hfrnn import torch_models as TM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"appareil : {dev} | N = {a.n} | {a.seeds} seeds | {a.steps} pas", flush=True)
    data = {t: tasks.load(t) for t in a.tasks}
    todo = [(s, t, l) for s in range(a.seeds) for t in a.tasks for l, _, _ in MODELS
            if a.only is None or any(w in l for w in a.only)]
    t0, done = time.time(), 0
    for seed, task, label in todo:
        cell = res["results"].setdefault(task, {}).setdefault(label, {})
        if str(seed) in cell:
            continue
        kw = dict(next(m for l, m, _ in MODELS if l == label))
        sparsity = kw.pop("sparsity", core.SPARSITY)
        Win, W = TM.make_reservoir(a.n, rho=core.RHO, in_scale=core.IN_SCALE, sparsity=sparsity, seed=seed)
        u, y = data[task]
        score, info = TM.train_regression(u, y, Win, W, seed, steps=a.steps, device=dev, **kw)
        cell[str(seed)] = {"nmse": score, "n_params": info["n_params"]}
        done += 1
        left = sum(1 for s, t, l in todo if str(s) not in res["results"].get(t, {}).get(l, {}))
        el = time.time() - t0
        print(f"[{el / 60:6.1f} min, reste ≈ {el / done * left / 60:5.0f} min] seed {seed} {task:20s} {label:34s} "
              f"NMSE = {score:.3g}", flush=True)
        with open(path, "w") as f:
            json.dump(res, f, indent=1)


def sc(res, t, l, seeds):
    c = res["results"][t].get(l, {})
    return np.array([c[str(s)]["nmse"] for s in range(seeds) if str(s) in c])


def report(res, tag):
    seeds = min(len(v) for t in res["results"] for v in res["results"][t].values() if v)
    if seeds == 0:
        print("Pas encore de seed complète."); return
    out = {}
    for t in TASKS:
        if t not in res["results"]:
            continue
        R = {l: sc(res, t, l, seeds) for l, _, _ in MODELS if len(sc(res, t, l, seeds)) == seeds}
        best = min([k for k in RNNS if k in R], key=lambda k: np.mean(np.log10(R[k])))
        print(f"\n{t} — N = {res['args']['n']}, {seeds} seeds. Écarts appariés log10 : au {REF_LABEL} ({best}) "
              f"et au réservoir accordé")
        out[t] = {"meilleur RNN": best, "modèles": {}}
        for l in R:
            d1 = np.log10(R[l] / R[best]); d2 = np.log10(R[l] / R[REF]) if REF in R else np.full(seeds, np.nan)
            se = lambda d: d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")   # noqa: E731
            npar = res["results"][t][l]["0"]["n_params"]
            print(f"  {l:34s} {npar:7d} param.  NMSE {R[l].mean():9.3g}   vs réf. {d1.mean():+.2f} ± {se(d1):.2f} "
                  f"(×{10 ** -d1.mean():4.1f})   vs accordé {d2.mean():+.2f} ± {se(d2):.2f}")
            out[t]["modèles"][l] = {"nmse": R[l].tolist(), "paramètres": npar, "vs RNN": float(d1.mean()),
                                    "vs RNN se": se(d1), "vs accordé": float(d2.mean()), "vs accordé se": se(d2)}
    with open(core.out_path("data", f"{PREFIX}_{tag}_summary.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    fig, axes = plt.subplots(1, len(out), figsize=(6.6 * len(out), 6.2))
    for ax, t in zip(np.atleast_1d(axes), out):
        labs = [l for l, _, _ in MODELS if l in out[t]["modèles"]]
        m = [np.mean(out[t]["modèles"][l]["nmse"]) for l in labs]
        se = [np.std(out[t]["modèles"][l]["nmse"], ddof=1) / np.sqrt(seeds) if seeds > 1 else 0 for l in labs]
        cols = [c for l, _, c in MODELS if l in labs]
        ax.bar(range(len(labs)), m, yerr=se, capsize=3, color=cols,
               edgecolor=["black" if l in NEW else "none" for l in labs])
        ax.axhline(np.mean(out[t]["modèles"][out[t]["meilleur RNN"]]["nmse"]), color="tab:red", ls="--", lw=1)
        ax.axhline(np.mean(out[t]["modèles"][REF]["nmse"]), color="tab:blue", ls=":", lw=1.2)
        ax.set_xticks(range(len(labs))); ax.set_xticklabels(labs, rotation=55, ha="right", fontsize=7.5)
        ax.set_yscale("log"); ax.set_ylabel("erreur de test (NMSE, log)")
        ax.set_title(f"{t}, N = {res['args']['n']}, {seeds} seeds", fontsize=11)
    fig.suptitle(TITLE, fontsize=11)
    fig.tight_layout(); fig.savefig(core.out_path("figures", f"{PREFIX}_{tag}.png"), dpi=110)
    print(f"\nFigure : results/figures/{PREFIX}_{tag}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=core.STEPS)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    tag = f"n{a.n}"
    path = core.out_path("data", f"{PREFIX}_{tag}.json")
    res = {"args": {"n": a.n, "steps": a.steps}, "results": {}}
    if path.exists():
        old = json.load(open(path))
        if old["args"] == res["args"]:
            res = old
    if not a.report:
        run(a, path, res)
    report(res, tag)


if __name__ == "__main__":
    main()
