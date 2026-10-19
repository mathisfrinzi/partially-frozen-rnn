"""
Partie 9, étape 1 — le gradient au départ contient-il l'information qui permet de choisir la structure ?

Pour chaque tâche (cinq) et chaque graine (trois), au réservoir de départ (100 neurones, lecture ridge), on calcule le
gradient de la perte sur 128 fenêtres d'entraînement et 128 fenêtres de validation, puis, pour chacune des treize structures
de la vérité terrain (parties 3 à 6, plus les bandes et le réservoir à biais et entrées appris), les signaux :

    part captée                  ‖P_s(ḡ)‖² / ‖ḡ‖²                       part de la descente que la structure peut suivre
    part par paramètre           idem, divisée par la dimension
    signal sur bruit             ‖P_s(ḡ)‖² / variance de la moyenne       direction stable d'un lot à l'autre ?
    accord entraînement-valid.   cos(P_s ḡ_entraînement, P_s ḡ_validation) ce qu'elle apprendra se généralise-t-il ?
    pas de Newton                ḡᵀ (F_s + λI)⁻¹ ḡ                        baisse prédite avec la courbure (Fisher)
    directions principales       part des 10 premières directions principales des gradients dans la structure
    mouvement naturel            ‖P_s(ΔW)‖² / ‖ΔW‖², ΔW = RNN entraîné − départ   (coûte un entraînement du RNN)
    entraînement pilote          −log10 NMSE de validation après 100 pas    (référence chère)
    taille                       dimension de la structure                 (référence naïve)
    choix naïf                   classement moyen des autres tâches         (référence, laisse une tâche de côté)

Chaque signal est comparé au gain réellement mesuré (log10 NMSE de test après 1 500 pas) : corrélation de rang entre
structures pour chaque tâche et chaque graine, et structure choisie par tâche (regret par rapport à la meilleure).

    python experiments/44_structure_signals.py            # CPU, ~30 min
    python experiments/44_structure_signals.py --report
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
N, B, R_PCS, PILOT = 100, 128, 10, 100
GRAD_SIGNALS = ["part captée", "part par paramètre", "signal sur bruit", "accord entraînement-validation",
                "pas de Newton", "directions principales"]
ALL_SIGNALS = GRAD_SIGNALS + ["mouvement naturel", "entraînement pilote", "taille", "choix naïf"]


def data(name):
    spec = importlib.util.spec_from_file_location("dec36", Path(__file__).with_name("36_decompositions.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.data(name)


def ground_truth():
    """log10 NMSE de test, {tâche: {structure: {graine: valeur}}}, à partir des fichiers de la partie 5 à 100 neurones."""
    from hfrnn.structure_signals import STRUCTURES
    names = [s for s, _ in STRUCTURES]
    out = {}
    for f in sorted(glob.glob(str(core.out_path("data", "decompositions_n100*.json")))):
        if f.endswith("_summary.json"):
            continue
        for t, R in json.load(open(f))["results"].items():
            for lab, cells in R.items():
                if lab in names:
                    for s, v in cells.items():
                        out.setdefault(t, {}).setdefault(lab, {})[s] = float(np.log10(v["nmse"]))
    return out


def run(a, path):
    import torch
    from hfrnn import structure_signals as SS
    from hfrnn import torch_models as TM
    torch.set_num_threads(a.threads)
    res = json.load(open(path)) if path.exists() else {"args": {"n": N, "B": B}, "signaux": {}}
    t0 = time.time()
    for seed in range(a.seeds):
        bases = None
        for task in (a.tasks or TASKS):
            cell = res["signaux"].setdefault(task, {}).setdefault(str(seed), {})
            u, y, nt = data(task)
            d = 1 if u.ndim == 1 else u.shape[1]
            Win, W = SS.reservoir(N, d, seed)
            if not all(s in cell and all(k in cell[s] for k in GRAD_SIGNALS) for s, _ in SS.STRUCTURES):
                if bases is None:
                    bases = {}
                    for s, kw in SS.STRUCTURES:
                        bases[s] = SS.orthonormal_basis(SS.tangent_jacobian(Win, W, seed, kw))
                        gc.collect()
                    print(f"[{(time.time() - t0) / 60:5.1f} min] graine {seed} : espaces tangents calculés", flush=True)
                w, c = SS.ridge_readout(u, y, nt, Win, W)
                nv = int(0.75 * nt)                                     # entraînement / validation, comme en 8.2
                rng = np.random.default_rng(500 + seed)
                tr = rng.integers(0, nv - core.WIN_LEN - 1, B)
                va = rng.integers(nv, nt - core.WIN_LEN - 1, B)
                gtr = SS.window_grads(u, y, tr, Win, W, w, c)
                gva = SS.window_grads(u, y, va, Win, W, w, c)
                Gtr = np.concatenate(gtr, 1)
                lam = float((Gtr ** 2).sum(1).mean() / Gtr.shape[1])     # valeur propre moyenne de la Fisher complète
                _, _, Vt = np.linalg.svd(Gtr, full_matrices=False)
                pcs = Vt[:R_PCS]
                nW, nI = N * N, Win.size
                for s, _ in SS.STRUCTURES:
                    Ztr = SS.coords(*gtr, bases[s])
                    Zva = SS.coords(*gva, bases[s])
                    P = SS.coords(pcs[:, :nW], pcs[:, nW:nW + nI], pcs[:, nW + nI:], bases[s])
                    cell.setdefault(s, {}).update(SS.signals(Ztr, Zva, Gtr, lam, P))
                print(f"[{(time.time() - t0) / 60:5.1f} min] {task:20s} graine {seed} : signaux du gradient", flush=True)
                json.dump(res, open(path, "w"), indent=1, ensure_ascii=False)
            # mouvement naturel : RNN entraîné comme dans la vérité terrain, ΔW projeté sur chaque structure
            if not all("mouvement naturel" in cell.get(s, {}) for s, _ in SS.STRUCTURES):
                if bases is None:
                    bases = {s: SS.orthonormal_basis(SS.tangent_jacobian(Win, W, seed, kw)) for s, kw in SS.STRUCTURES}
                _, info = TM.train_regression(u, y, Win, W, seed, kind="convex", alpha=1.0, n_train=nt,
                                              return_model=True, steps=a.rnn_steps)
                Wi, Wt = info["model"].weights()
                dW = (Wt.detach().numpy() - W).reshape(1, -1)
                dI = (Wi.detach().numpy() - Win).reshape(1, -1)
                db = info["model"].bias.detach().numpy().reshape(1, -1)
                tot = (dW ** 2).sum() + (dI ** 2).sum() + (db ** 2).sum()
                for s, _ in SS.STRUCTURES:
                    z = SS.coords(dW, dI, db, bases[s])
                    cell.setdefault(s, {})["mouvement naturel"] = float((z ** 2).sum() / tot)
                print(f"[{(time.time() - t0) / 60:5.1f} min] {task:20s} graine {seed} : mouvement naturel", flush=True)
                json.dump(res, open(path, "w"), indent=1, ensure_ascii=False)
            # entraînement pilote : 100 pas sur les 75 % premiers de l'entraînement, NMSE sur les 25 % suivants
            for s, kw in SS.STRUCTURES:
                if "entraînement pilote" in cell.get(s, {}):
                    continue
                v, _ = TM.train_regression(u[:nt], y[:nt], Win, W, seed, steps=a.pilot, **dict(kw))
                cell.setdefault(s, {})["entraînement pilote"] = float(-np.log10(v))
                json.dump(res, open(path, "w"), indent=1, ensure_ascii=False)
            print(f"[{(time.time() - t0) / 60:5.1f} min] {task:20s} graine {seed} : entraînement pilote", flush=True)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def evaluate(res, gt):
    """Pour chaque signal : corrélation de rang (moyenne ± erreur type sur tâche × graine), et structure choisie par tâche."""
    S = res["signaux"]
    tasks = [t for t in TASKS if t in S and t in gt]
    names = sorted(set.intersection(*[set(gt[t]) for t in tasks]) & set.intersection(*[set(S[t]["0"]) for t in tasks]),
                   key=lambda s: list(S[tasks[0]]["0"]).index(s))
    seeds = sorted(set.intersection(*[set(S[t]) for t in tasks]), key=int)
    perf = {t: {s: np.array([gt[t][s][k] for k in seeds]) for s in names} for t in tasks}   # log10 NMSE
    out = {"structures": names, "tâches": tasks, "graines": len(seeds), "signaux": {}}
    for sig in ALL_SIGNALS:
        rhos, picks = [], {}
        for t in tasks:
            if sig == "choix naïf":                                   # classement moyen des autres tâches
                score = {s: -np.mean([perf[o][s].mean() for o in tasks if o != t]) for s in names}
                vals = [np.array([score[s]] * len(seeds)) for s in names]
            elif sig == "taille":
                vals = [np.array([S[t][k][s]["dimension"] for k in seeds]) for s in names]
            else:
                if not all(sig in S[t][k][s] for k in seeds for s in names):
                    break
                vals = [np.array([S[t][k][s][sig] for k in seeds]) for s in names]
            V = np.array(vals)                                          # (structures, graines)
            for j in range(len(seeds)):
                rhos.append(spearman(V[:, j], -np.array([perf[t][s][j] for s in names])))
            ranks = np.argsort(np.argsort(V, 0), 0).mean(1)               # rang moyen sur les graines
            chosen = names[int(np.argmax(ranks))]
            best = min(names, key=lambda s: perf[t][s].mean())
            picks[t] = {"choisie": chosen, "meilleure": best,
                        "regret": float(perf[t][chosen].mean() - perf[t][best].mean())}
        else:
            r = np.array(rhos)
            out["signaux"][sig] = {"rho": float(r.mean()), "se": float(r.std(ddof=1) / np.sqrt(len(r))), "choix": picks,
                                   "regret moyen": float(np.mean([p["regret"] for p in picks.values()])),
                                   "bons choix": int(sum(p["choisie"] == p["meilleure"] for p in picks.values()))}
    # information propre à la tâche : signal et gain doublement centrés (moins la moyenne de chaque structure et de chaque
    # tâche), corrélés ; p par permutation des structures à l'intérieur de chaque tâche
    Pm = np.array([[-perf[t][s].mean() for s in names] for t in tasks])
    dc = lambda M: M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()   # noqa: E731
    rng = np.random.default_rng(0)
    for sig in GRAD_SIGNALS + ["mouvement naturel", "entraînement pilote"]:
        if sig not in out["signaux"]:
            continue
        V = np.array([[np.mean([S[t][k][s][sig] for k in seeds]) for s in names] for t in tasks])
        X = V if sig in ("accord entraînement-validation", "entraînement pilote") else np.log(np.abs(V) + 1e-300)
        r = float(np.corrcoef(dc(X).ravel(), dc(Pm).ravel())[0, 1])
        null = [np.corrcoef(dc(np.array([row[rng.permutation(len(names))] for row in X])).ravel(), dc(Pm).ravel())[0, 1]
                for _ in range(2000)]
        out["signaux"][sig]["interaction"] = {"r": r, "p": float(np.mean(np.abs(null) >= abs(r)))}
    # référence : regret moyen d'un choix au hasard
    out["hasard, regret moyen"] = float(np.mean([np.mean([perf[t][s].mean() for s in names])
                                                 - min(perf[t][s].mean() for s in names) for t in tasks]))
    return out, perf


def report(path):
    from hfrnn.structure_signals import SHORT
    res = json.load(open(path))
    gt = ground_truth()
    out, perf = evaluate(res, gt)
    print(f"\n{len(out['structures'])} structures, {len(out['tâches'])} tâches, {out['graines']} graines. Corrélation de rang "
          f"entre le signal et le gain mesuré (1 = classement parfait, 0 = aucun lien), structure choisie par tâche.")
    print(f"  {'signal':32s} {'rho':>12s}  {'bons choix':>10s}  {'regret moyen (log10)':>20s}")
    for sig, v in out["signaux"].items():
        print(f"  {sig:32s} {v['rho']:+.2f} ± {v['se']:.2f}  {v['bons choix']:>6d} / {len(out['tâches'])}  {v['regret moyen']:>12.2f}")
    print(f"  {'hasard':32s} {'0':>12s}  {'':>10s}  {out['hasard, regret moyen']:>12.2f}")
    print("\nInformation propre à la tâche (signal et gain doublement centrés ; p par permutation) :")
    for sig, v in out["signaux"].items():
        if "interaction" in v:
            print(f"  {sig:32s} r = {v['interaction']['r']:+.2f}   p = {v['interaction']['p']:.3f}")
    for sig, v in out["signaux"].items():
        print(f"\n  {sig} : " + " ; ".join(f"{t[:10]} → {SHORT.get(p['choisie'], p['choisie'])} "
                                          f"(meilleure : {SHORT.get(p['meilleure'], p['meilleure'])}, regret {p['regret']:+.2f})"
                                          for t, p in v["choix"].items()))
    json.dump(out, open(core.out_path("data", "structure_signals_summary.json"), "w"), indent=1, ensure_ascii=False)
    figures(res, out, perf)


def figures(res, out, perf):
    from hfrnn.structure_signals import SHORT
    sigs = list(out["signaux"])
    # 1. corrélation de rang par signal, et par tâche
    S = res["signaux"]
    tasks, names = out["tâches"], out["structures"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    y = np.arange(len(sigs))
    ax[0].errorbar([out["signaux"][s]["rho"] for s in sigs], y, xerr=[2 * out["signaux"][s]["se"] for s in sigs],
                   fmt="o", color="tab:blue", capsize=3)
    ax[0].axvline(0, color="gray", lw=0.8)
    ax[0].set_yticks(y); ax[0].set_yticklabels(sigs, fontsize=9); ax[0].invert_yaxis()
    ax[0].set_xlabel("corrélation de rang avec le gain mesuré (± 2 erreurs types)"); ax[0].set_xlim(-1, 1)
    ax[0].set_title("Le signal classe-t-il les structures comme l'entraînement ?", fontsize=10)
    ax[1].barh(y, [out["signaux"][s]["regret moyen"] for s in sigs], color="tab:orange")
    ax[1].axvline(out["hasard, regret moyen"], color="gray", ls="--", label="choix au hasard")
    ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].invert_yaxis()
    ax[1].set_xlabel("regret moyen de la structure choisie (log10 NMSE ; 0 = la meilleure)"); ax[1].legend(fontsize=8)
    ax[1].set_title("Structure choisie par tâche", fontsize=10)
    fig.tight_layout(); fig.savefig(core.out_path("figures", "structure_signals.png"), dpi=110)
    # 2. le meilleur signal du gradient contre le gain mesuré, tâche par tâche
    best = max([s for s in sigs if s in GRAD_SIGNALS], key=lambda s: out["signaux"][s]["interaction"]["r"])
    fig, axes = plt.subplots(1, len(tasks), figsize=(3.4 * len(tasks), 3.4), squeeze=False)
    for axx, t in zip(axes[0], tasks):
        xs = [np.mean([S[t][k][s][best] for k in S[t]]) for s in names]
        ys = [perf[t][s].mean() for s in names]
        axx.scatter(xs, ys, color="tab:blue", s=18)
        for s, xv, yv in zip(names, xs, ys):
            axx.annotate(SHORT.get(s, s), (xv, yv), fontsize=6, xytext=(2, 2), textcoords="offset points")
        if min(xs) > 0:
            axx.set_xscale("log")
        axx.set_title(t, fontsize=9); axx.set_xlabel(best, fontsize=8)
    axes[0, 0].set_ylabel("log10 NMSE de test après entraînement")
    fig.suptitle(f"Le signal du gradient le plus lié à la tâche ({best}) contre l'erreur après entraînement", fontsize=10)
    fig.tight_layout(); fig.savefig(core.out_path("figures", "structure_signals_scatter.png"), dpi=110)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--report", action="store_true")
    p.add_argument("--tasks", nargs="*")
    p.add_argument("--quick", action="store_true", help="vérification rapide (16 neurones, quelques pas)")
    a = p.parse_args()
    a.rnn_steps, a.pilot = core.STEPS, PILOT
    if a.quick:
        N, B, a.rnn_steps, a.pilot, a.seeds = 16, 16, 10, 5, 1
    path = core.out_path("data", f"structure_signals_n{N}.json")
    if not a.report:
        run(a, path)
    report(path)
