"""
Partie 9, étape 2 — répéter le sondage en cours d'entraînement.

À l'étape 1, les signaux du gradient étaient mesurés une seule fois, au réservoir de départ. Ici le RNN s'entraîne
t pas (t = 0, 30, 100, 300, 1000) ; à chaque t on se pose la question de nouveau, à partir de l'état atteint (W_t, W_in,t, b_t) :
quelle structure apprendre pour la suite ? Chaque structure repart de W_t (sa décomposition est refaite sur W_t).

    signaux    les six signaux du gradient de l'étape 1, mesurés en W_t sur 128 fenêtres d'entraînement et 128 de validation,
               pour cinq structures : RNN, accordé, tourné, cisaillé (bande k = 10), biais et entrées seuls
    référence  l'entraînement pilote (100 pas depuis W_t) et la taille
    mesure     log10 NMSE de test après 500 pas d'entraînement de chaque structure depuis W_t (« bascule » au pas t)

On regarde si le signal choisit mieux la structure plus tard dans l'entraînement qu'au départ (bons choix, regret), et
si la meilleure structure change avec t.

    python experiments/45_signaux_dynamiques.py --seed 0            # une graine (~25 min CPU), écrit signaux_dyn_s0.json
    python experiments/45_signaux_dynamiques.py --report            # fusionne les graines et résume
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
CHECK = [0, 30, 100, 300, 1000]
NAMES = ["RNN", "réservoir accordé (valeurs singulières)", "réservoir tourné (rotations)",
         "réservoir cisaillé en bande, k = 10", "réservoir à biais et entrées appris (sans g)"]
FAMILIES = NAMES[1:4]                     # tourner, accorder, cisailler
SHORT = {NAMES[0]: "RNN", NAMES[1]: "accordé", NAMES[2]: "tourné", NAMES[3]: "cisaillé", NAMES[4]: "biais et entrées"}
GRAD_SIGNALS = ["part captée", "part par paramètre", "signal sur bruit", "accord entraînement-validation",
                "pas de Newton", "directions principales"]
SIGNALS = GRAD_SIGNALS + ["entraînement pilote", "taille"]


def data(name):
    spec = importlib.util.spec_from_file_location("dec36", Path(__file__).with_name("36_decompositions.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.data(name)


def structures():
    from hfrnn.structure_signals import STRUCTURES
    d = dict(STRUCTURES)
    return [(s, d[s]) for s in NAMES]


def state_at(u, y, nt, Win0, W0, seed, t, device):
    """(Win, W, b) du RNN après t pas d'entraînement (convexe, alpha = 1, comme la vérité terrain)."""
    import torch
    from hfrnn import torch_models as TM
    if t == 0:
        return Win0, W0, np.zeros(W0.shape[0])
    _, info = TM.train_regression(u, y, Win0, W0, seed, kind="convex", alpha=1.0, steps=t, n_train=nt, device=device,
                                  return_model=True)
    m = info["model"]
    with torch.no_grad():
        Wi, Wt = m.weights()
    return Wi.cpu().double().numpy(), Wt.cpu().double().numpy(), m.bias.detach().cpu().double().numpy()


def switch(u, y, nt, Win, W, b, seed, kw, steps, device, valid=False):
    """Entraîne la structure `kw` à partir de l'état (Win, W, b) ; log10 NMSE de test (ou de validation si `valid`)."""
    import torch
    from hfrnn import torch_models as TM
    model = TM.build_model(Win, W, kw["kind"], seed, torch.float32, device, **{k: v for k, v in kw.items() if k != "kind"})
    with torch.no_grad():
        model.bias.copy_(torch.as_tensor(b, dtype=torch.float32, device=device))
    if valid:
        v, _ = TM.train_regression(u[:nt], y[:nt], Win, W, seed, model=model, steps=steps, device=device)
    else:
        v, _ = TM.train_regression(u, y, Win, W, seed, model=model, steps=steps, device=device, n_train=nt)
    return float(np.log10(v))


def run(a, path):
    import torch
    from hfrnn import structure_signals as SS
    torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    res = json.load(open(path)) if path.exists() else {"args": {"n": a.n, "switch": a.switch, "pilot": a.pilot}, "cells": {}}
    t0 = time.time()
    S = structures()
    for task in (a.tasks or TASKS):
        u, y, nt = data(task)
        d = 1 if u.ndim == 1 else u.shape[1]
        Win0, W0 = SS.reservoir(a.n, d, a.seed)
        for t in a.check:
            cell = res["cells"].setdefault(task, {}).setdefault(str(t), {})
            if all(s in cell and "gain" in cell[s] and "entraînement pilote" in cell[s] for s in NAMES):
                continue
            Win, W, b = state_at(u, y, nt, Win0, W0, a.seed, t, device)
            sig = SS.probe(u, y, nt, Win, W, b, a.seed, S, B=a.windows)
            for s, kw in S:
                c = cell.setdefault(s, {})
                c.update(sig[s])
                c["gain"] = switch(u, y, nt, Win, W, b, a.seed, kw, a.switch, device)
                c["entraînement pilote"] = -switch(u, y, nt, Win, W, b, a.seed, kw, a.pilot, device, valid=True)
            json.dump(res, open(path, "w"), indent=1, ensure_ascii=False)
            print(f"[{(time.time() - t0) / 60:5.1f} min] {task:20s} graine {a.seed}  t = {t:4d}", flush=True)


# ------------------------------------------------------------------------------------------------ évaluation
def load(pattern):
    """Fusionne les fichiers de graines : {tâche: {t: {graine: {structure: {signaux, gain}}}}}."""
    out = {}
    for f in sorted(glob.glob(pattern)):
        r = json.load(open(f))
        seed = Path(f).stem.rsplit("_s", 1)[1]
        for task, T in r["cells"].items():
            for t, cell in T.items():
                if all(s in cell and "gain" in cell[s] for s in NAMES):
                    out.setdefault(task, {}).setdefault(int(t), {})[seed] = cell
    return out


def score(cells, task, t, sig, names, tasks):
    """Valeur du signal pour chaque structure : tableau (structures, graines)."""
    seeds = sorted(cells[task][t])
    if sig == "taille":
        return np.array([[cells[task][t][k][s]["dimension"] for k in seeds] for s in names], float)
    return np.array([[cells[task][t][k][s][sig] for k in seeds] for s in names], float)


def choose(cells, tasks, t, sig, names):
    """Structure choisie par le signal pour chaque tâche (rang moyen sur les graines), et regret par rapport à la meilleure."""
    picks = {}
    for task in tasks:
        seeds = sorted(cells[task][t])
        V = score(cells, task, t, sig, names, tasks)
        ranks = np.argsort(np.argsort(V, 0), 0).mean(1)
        gain = np.array([np.mean([cells[task][t][k][s]["gain"] for k in seeds]) for s in names])
        i, j = int(np.argmax(ranks)), int(np.argmin(gain))
        picks[task] = {"choisie": names[i], "meilleure": names[j], "regret": float(gain[i] - gain[j])}
    return picks


def interaction(X, P, rng, n_perm=2000):
    """Corrélation des matrices doublement centrées (tâche × structure) du signal et du gain, p par permutation."""
    dc = lambda M: M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()   # noqa: E731
    r = float(np.corrcoef(dc(X).ravel(), dc(P).ravel())[0, 1])
    null = [np.corrcoef(dc(np.array([row[rng.permutation(len(row))] for row in X])).ravel(), dc(P).ravel())[0, 1]
            for _ in range(n_perm)]
    return r, float(np.mean(np.abs(null) >= abs(r)))


def evaluate(cells):
    full = max(len(cells[t]) for t in cells)                         # tâches dont tous les pas t sont calculés
    tasks = [t for t in TASKS if t in cells and len(cells[t]) == full]
    left = [t for t in TASKS if t in cells and t not in tasks]
    if left:
        print("tâches incomplètes, laissées de côté :", ", ".join(f"{t} (t = {sorted(cells[t])})" for t in left))
    checks = sorted(set.intersection(*[set(cells[t]) for t in tasks]))
    out = {"tâches": tasks, "t": checks, "graines": min(len(cells[t][c]) for t in tasks for c in checks),
           "meilleure": {}, "choix": {}}
    rng = np.random.default_rng(0)
    for t in checks:
        out["meilleure"][t] = {}
        for task in tasks:
            seeds = sorted(cells[task][t])
            gain = {s: np.mean([cells[task][t][k][s]["gain"] for k in seeds]) for s in NAMES}
            out["meilleure"][t][task] = {SHORT[s]: gain[s] for s in NAMES}
        for sig in SIGNALS:
            for ens, names in (("3 familles", FAMILIES), ("5 structures", NAMES)):
                p = choose(cells, tasks, t, sig, names)
                row = {"bons": sum(v["choisie"] == v["meilleure"] for v in p.values()),
                       "regret": float(np.mean([v["regret"] for v in p.values()])), "choix": p}
                if ens == "5 structures" and sig in GRAD_SIGNALS + ["entraînement pilote"]:
                    X = np.array([[np.mean([cells[task][t][k][s][sig] for k in cells[task][t]]) for s in NAMES]
                                  for task in tasks])
                    if sig not in ("accord entraînement-validation", "entraînement pilote"):
                        X = np.log(np.abs(X) + 1e-300)
                    P = np.array([[-np.mean([cells[task][t][k][s]["gain"] for k in cells[task][t]]) for s in NAMES]
                                  for task in tasks])
                    row["r"], row["p"] = interaction(X, P, rng)
                out["choix"].setdefault(t, {}).setdefault(sig, {})[ens] = row
        # hasard : regret moyen d'un choix uniforme
        for ens, names in (("3 familles", FAMILIES), ("5 structures", NAMES)):
            reg = []
            for task in tasks:
                g = [np.mean([cells[task][t][k][s]["gain"] for k in cells[task][t]]) for s in names]
                reg.append(np.mean(g) - min(g))
            out["choix"][t].setdefault("hasard", {})[ens] = {"regret": float(np.mean(reg)), "bons": len(tasks) / len(names)}
    return out


def report(a):
    cells = load(str(core.out_path("data", "signaux_dyn_s[0-9]*.json")))
    out = evaluate(cells)
    tasks, checks = out["tâches"], out["t"]
    print(f"\n{len(tasks)} tâches, {out['graines']} graines, pas t = {checks}. Gain = log10 NMSE de test après la bascule.")
    print("\nMeilleure structure par tâche selon t (gain en log10 NMSE) :")
    for t in checks:
        print(f"  t = {t:4d} : " + " ; ".join(
            f"{task[:10]} → {min(out['meilleure'][t][task], key=out['meilleure'][t][task].get)}" for task in tasks))
    for ens in ("3 familles", "5 structures"):
        print(f"\nChoix parmi {ens} : bons choix sur {len(tasks)} tâches / regret moyen (log10), par pas t")
        print(f"  {'signal':32s}" + "".join(f"{('t=' + str(t)):>14s}" for t in checks))
        for sig in SIGNALS + ["hasard"]:
            row = out["choix"][checks[0]][sig]
            if ens not in row:
                continue
            print(f"  {sig:32s}" + "".join(
                f"{out['choix'][t][sig][ens]['bons']:>8.0f} / {out['choix'][t][sig][ens]['regret']:.2f}" for t in checks))
    print("\nInformation propre à la tâche (r doublement centré, p par permutation), parmi les 5 structures :")
    print(f"  {'signal':32s}" + "".join(f"{('t=' + str(t)):>14s}" for t in checks))
    for sig in GRAD_SIGNALS + ["entraînement pilote"]:
        print(f"  {sig:32s}" + "".join(
            f"{out['choix'][t][sig]['5 structures']['r']:>+7.2f} p={out['choix'][t][sig]['5 structures']['p']:.2f}" for t in checks))
    json.dump(out, open(core.out_path("data", "signaux_dyn_summary.json"), "w"), indent=1, ensure_ascii=False, default=str)
    figures(cells, out)


def figures(cells, out):
    tasks, checks = out["tâches"], out["t"]
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    for sig in SIGNALS + ["hasard"]:
        ax[0].plot(range(len(checks)), [out["choix"][t][sig]["3 familles"]["regret"] for t in checks], marker="o",
                   lw=2 if sig in ("signal sur bruit", "entraînement pilote", "hasard") else 1,
                   ls="--" if sig == "hasard" else "-", label=sig)
    ax[0].set_xticks(range(len(checks))); ax[0].set_xticklabels(checks)
    ax[0].set_xlabel("pas d'entraînement du RNN avant la bascule (t)")
    ax[0].set_ylabel("regret moyen du choix (log10 NMSE ; 0 = la meilleure)")
    ax[0].set_title("Choisir entre accordé, tourné et cisaillé", fontsize=10); ax[0].legend(fontsize=7)
    for task in tasks:
        v = [np.mean([np.log10(cells[task][t][k][NAMES[3]]["signal sur bruit"]) for k in cells[task][t]]) for t in checks]
        ax[1].plot(range(len(checks)), v, marker="o", label=task)
    ax[1].set_xticks(range(len(checks))); ax[1].set_xticklabels(checks)
    ax[1].set_xlabel("pas d'entraînement du RNN (t)"); ax[1].set_ylabel("log10 du signal sur bruit (cisaillé, k = 10)")
    ax[1].set_title("Le signal sur bruit change-t-il avec l'entraînement ?", fontsize=10); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(core.out_path("figures", "signaux_dynamiques.png"), dpi=110)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--windows", type=int, default=128)
    p.add_argument("--switch", type=int, default=500, help="pas d'entraînement après la bascule")
    p.add_argument("--pilot", type=int, default=100)
    p.add_argument("--check", type=int, nargs="*", default=CHECK)
    p.add_argument("--tasks", nargs="*")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--quick", action="store_true", help="vérification rapide (16 neurones, quelques pas)")
    a = p.parse_args()
    if a.quick:
        a.n, a.windows, a.switch, a.pilot, a.check = 16, 16, 8, 4, [0, 6]
    if not a.report:
        run(a, core.out_path("data", f"signaux_dyn_s{a.seed}.json"))
    else:
        report(a)
