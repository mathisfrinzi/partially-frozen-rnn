"""
Partie 8.2 — Comparaison avec des modèles existants (GPU) : GRU, LSTM, LRU contre nos meilleurs modèles.

Même protocole pour tous (fenêtres, Adam, clipping, readout ridge final), même taille d'état N.
Entraînement plus long que dans le reste de l'étude (4 000 pas au lieu de 1 500), avec la NMSE mesurée
en cours de route : on voit si les modèles classiques rattrapent leur retard quand on les laisse apprendre.
Les modèles existants et le RNN ont deux learning rates chacun ; on garde le meilleur par tâche (choisi sur
le test, ce qui les avantage).

    python experiments/12_comparisons.py --quick            # vérification (~3 min)
    python experiments/12_comparisons.py                    # N = 500, 3 seeds, 3 tâches
    python experiments/12_comparisons.py --n 200 --seeds 5
    python experiments/12_comparisons.py --report           # tableau et figures seulement (sans GPU)

Les résultats sont sauvegardés après chaque entraînement ; une relance reprend là où le calcul s'était arrêté.
Ordre : seed par seed, donc un calcul interrompu donne déjà une comparaison complète sur les premières seeds.
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
TASKS = ["Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)"]
DENSE = {"sparsity": 1.0}
# (étiquette, famille, réglages)
MODELS = [
    ("réservoir classique", "réservoir classique", {"kind": "reservoir"}),
    ("RNN, lr × 0.1", "RNN", {"kind": "convex", "alpha": 1.0, "lr": 0.1 * LR}),
    ("RNN, lr × 0.03", "RNN", {"kind": "convex", "alpha": 1.0, "lr": 0.03 * LR}),
    ("GRU, lr 2e-3", "GRU", {"kind": "gru"}),
    ("GRU, lr 5e-4", "GRU", {"kind": "gru", "lr": 5e-4}),
    ("LSTM, lr 2e-3", "LSTM", {"kind": "lstm"}),
    ("LSTM, lr 5e-4", "LSTM", {"kind": "lstm", "lr": 5e-4}),
    ("LRU, lr 2e-3", "LRU", {"kind": "lru"}),
    ("LRU, lr 5e-4", "LRU", {"kind": "lru", "lr": 5e-4}),
    # réglages plus soignés des modèles existants (ajoutés après la première seed)
    ("GRU, init soignée, lr 2e-3", "GRU", {"kind": "gru", "rec_init": "ortho"}),
    ("GRU, init soignée, lr 5e-4", "GRU", {"kind": "gru", "rec_init": "ortho", "lr": 5e-4}),
    ("LSTM, init soignée, lr 2e-3", "LSTM", {"kind": "lstm", "rec_init": "ortho"}),
    ("LSTM, init soignée, lr 5e-4", "LSTM", {"kind": "lstm", "rec_init": "ortho", "lr": 5e-4}),
    ("LRU 2 couches, lr 2e-3", "LRU", {"kind": "lru", "layers": 2}),
    ("LRU 2 couches, lr 5e-4", "LRU", {"kind": "lru", "layers": 2, "lr": 5e-4}),
    ("RNN freiné", "RNN freiné", {"kind": "convex", "alpha": 1.0, "lr_mult": 0.1, "lr_mult_in": 1.0}),
    ("réservoir corrigé de rang 4", "réservoir corrigé de rang 4", {"kind": "lowrank", "rank": 4}),
    ("réservoir sculpté intégral, dense", "réservoir sculpté intégral",
     dict({"kind": "supermask", "mode": "sigmoid", "win": "mask", "mult": 50.0}, **DENSE)),
    ("réservoir accordé", "réservoir accordé", {"kind": "spectrum"}),
    # partie 6.3 : les meilleures variantes des parties 4.2 et 5, ajoutées après les trois premières seeds
    ("accordé sélectif", "accordé sélectif", {"kind": "tuned", "selective": True}),
    ("réservoir accordé à portes", "réservoir accordé à portes", {"kind": "gated_tuned", "cell": "gru"}),
    ("réservoir accordé à porte d'entrée", "réservoir accordé à porte d'entrée", {"kind": "gated_tuned", "cell": "mingru"}),
]
FAMILY = {l: f for l, f, _ in MODELS}
EXISTING = ["RNN", "GRU", "LSTM", "LRU"]
OURS = ["RNN freiné", "réservoir corrigé de rang 4", "réservoir sculpté intégral", "réservoir accordé",
        "accordé sélectif", "réservoir accordé à portes", "réservoir accordé à porte d'entrée"]
COLORS = {"réservoir classique": "gray", "RNN": "tab:red", "GRU": "tab:orange", "LSTM": "goldenrod",
          "LRU": "tab:pink", "RNN freiné": "tab:brown", "réservoir corrigé de rang 4": "tab:olive",
          "réservoir sculpté intégral": "tab:purple", "réservoir accordé": "tab:blue",
          "accordé sélectif": "lightsteelblue", "réservoir accordé à portes": "tab:cyan",
          "réservoir accordé à porte d'entrée": "darkcyan"}


def run(a, path, res):
    import torch
    from hfrnn import torch_models as TM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"appareil : {dev} | N = {a.n} | {a.seeds} seeds | {a.steps} pas | points de mesure {a.eval_at}",
          flush=True)
    data = {t: tasks.load(t) for t in a.tasks}
    t0, done = time.time(), 0
    todo = [(s, t, l) for s in range(a.seeds) for t in a.tasks for l, _, _ in MODELS
            if a.only is None or any(w in l for w in a.only)]
    for seed, task, label in todo:
        cell = res["results"].setdefault(task, {}).setdefault(label, {})
        if str(seed) in cell:
            continue
        kw = dict(next(m for l, _, m in MODELS if l == label))
        sparsity = kw.pop("sparsity", core.SPARSITY)
        Win, W = TM.make_reservoir(a.n, rho=core.RHO, in_scale=core.IN_SCALE, sparsity=sparsity, seed=seed)
        u, y = data[task]
        t1 = time.time()
        score, info = TM.train_regression(u, y, Win, W, seed, steps=a.steps, device=dev,
                                          eval_at=tuple(a.eval_at), **kw)
        cell[str(seed)] = {"nmse": score, "checkpoints": {str(k): v for k, v in info["checkpoints"].items()},
                           "n_params": info["n_params"], "seconds": time.time() - t1}
        done += 1
        left = sum(1 for s, t, l in todo if str(s) not in res["results"].get(t, {}).get(l, {}))
        el = time.time() - t0
        print(f"[{el / 60:6.1f} min, reste ≈ {el / done * left / 60:5.0f} min] seed {seed} {task:20s} "
              f"{label:36s} NMSE = {score:.3g}  ({info['n_params']} paramètres, {time.time() - t1:.0f} s)",
              flush=True)
        with open(path, "w") as f:
            json.dump(res, f, indent=1)


def scores(res, task, label, step=None):
    """NMSE par seed (dans l'ordre des seeds), au pas `step` (défaut : fin)."""
    cell = res["results"][task].get(label, {})
    out = []
    for s in sorted(cell, key=int):
        if step is None:
            out.append(cell[s]["nmse"])
        else:                                   # le réservoir classique n'est pas entraîné : même score partout
            fallback = cell[s]["nmse"] if FAMILY[label] == "réservoir classique" else np.nan
            out.append(cell[s]["checkpoints"].get(str(step), fallback))
    return np.array(out, dtype=float)


def best_of_family(res, task, fam, seeds):
    """Le meilleur réglage d'une famille sur cette tâche (NMSE moyenne finale), parmi les seeds complètes."""
    labs = [l for l, f, _ in MODELS if f == fam and len(scores(res, task, l)) >= seeds]
    return min(labs, key=lambda l: np.mean(np.log10(scores(res, task, l)[:seeds]))) if labs else None


def report(res, tag):
    tasks_ = list(res["results"])
    # seeds complètes : celles où tous les modèles ont tourné, pour une comparaison appariée
    seeds = min(len(res["results"][t][l]) for t in tasks_ for l in res["results"][t] if res["results"][t][l])
    if seeds == 0:
        print("Pas encore de seed complète."); return
    steps = sorted({int(k) for t in tasks_ for l in res["results"][t] for c in res["results"][t][l].values()
                    for k in c["checkpoints"]} - {0})
    fams = ["réservoir classique"] + EXISTING + OURS
    summary = {}
    for t in tasks_:
        best = {f: best_of_family(res, t, f, seeds) for f in fams}
        fams_t = [f for f in fams if best[f] is not None]
        ref_fam = min([f for f in EXISTING if best[f]], key=lambda f: np.mean(np.log10(scores(res, t, best[f])[:seeds])))
        ref = scores(res, t, best[ref_fam])[:seeds]
        print(f"\n{t} — {seeds} seeds, {res['args']['steps']} pas. Référence : meilleur modèle existant = "
              f"{best[ref_fam]}")
        print(f"  {'modèle':36s} {'paramètres':>10s}  " + "  ".join(f"pas {s:>5d}" for s in steps)
              + "   × moins d'erreur que la référence (apparié)")
        summary[t] = {"référence": best[ref_fam], "modèles": {}}
        for f in fams_t:
            l = best[f]
            v = scores(res, t, l)[:seeds]
            d = np.log10(v / ref)
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            npar = res["results"][t][l]["0"]["n_params"]
            curve = [np.nanmean(scores(res, t, l, s)[:seeds]) for s in steps]
            print(f"  {l:36s} {npar:10d}  " + "  ".join(f"{c:9.3g}" for c in curve)
                  + f"   ×{10 ** -d.mean():.2f}  (écart {d.mean():+.2f} ± {se:.2f})")
            summary[t]["modèles"][f] = {"réglage": l, "paramètres": npar, "courbe": dict(zip(map(str, steps), curve)),
                                        "écart": float(d.mean()), "se": float(se), "nmse": v.tolist()}
    with open(core.out_path("data", f"comparisons_{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    fig, axes = plt.subplots(2, len(tasks_), figsize=(6.3 * len(tasks_), 10.5), squeeze=False)
    for j, t in enumerate(tasks_):
        ax = axes[0, j]
        m = summary[t]["modèles"]
        vals = [np.mean(m[f]["nmse"]) for f in fams]
        se = [np.std(m[f]["nmse"], ddof=1) / np.sqrt(seeds) if seeds > 1 else 0 for f in fams]
        ax.bar(range(len(fams)), vals, yerr=se, capsize=3, color=[COLORS[f] for f in fams])
        ax.axvline(len(EXISTING) + 0.5, color="black", lw=0.8)
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels([f"{f}\n({m[f]['paramètres'] // 1000} k)" if m[f]["paramètres"] >= 1000 else
                            f"{f}\n({m[f]['paramètres']})" for f in fams], rotation=40, ha="right", fontsize=8)
        ax.set_yscale("log"); ax.set_ylabel("erreur de test (NMSE, log)")
        ax.set_title(f"{t}, N = {res['args']['n']}, {res['args']['steps']} pas", fontsize=11)
        ax.text(0.02, 0.97, "existants  |  nos modèles →", transform=ax.transAxes, fontsize=8, va="top")
        ax = axes[1, j]
        for f in fams[1:]:
            ax.plot(steps, [m[f]["courbe"][str(s)] for s in steps], marker="o", color=COLORS[f], label=f,
                    ls="--" if f in EXISTING else "-")
        ax.set_xlabel("pas d'entraînement"); ax.set_ylabel("NMSE test (log)"); ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle("Partie 6 — modèles existants (tirets) contre nos modèles (traits pleins). Entre parenthèses : "
                 "paramètres récurrents appris.", fontsize=11)
    fig.tight_layout(); fig.savefig(core.out_path("figures", f"comparisons_{tag}.png"), dpi=110)
    print(f"\nFigure : results/figures/comparisons_{tag}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--eval-at", type=int, nargs="*", default=[1000, 2000, 3000])
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--only", nargs="*", default=None, help="ne lancer que les modèles dont le nom contient ces mots")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report", action="store_true", help="tableau et figures à partir des résultats sauvegardés")
    a = ap.parse_args()
    if a.quick:
        a.n, a.seeds, a.steps, a.eval_at = 64, 1, 60, [30]
    tag = f"n{a.n}" + ("_quick" if a.quick else "")
    path = core.out_path("data", f"comparisons_{tag}.json")
    res = {"args": {"n": a.n, "steps": a.steps, "eval_at": a.eval_at}, "results": {}}
    if path.exists():
        old = json.load(open(path))
        if old["args"]["n"] == a.n and old["args"]["steps"] == a.steps:
            res = old
            print("résultats existants repris", flush=True)
    if not a.report:
        run(a, path, res)
    report(res, tag)


if __name__ == "__main__":
    main()
