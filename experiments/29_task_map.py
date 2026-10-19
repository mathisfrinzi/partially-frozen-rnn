"""
Partie 8.3 — Carte des tâches : pour quelles données le réservoir accordé marche-t-il ?

Douze tâches synthétiques de familles différentes (chaos, identification de systèmes, canaux et amplificateurs non
linéaires, bruit, peu de données, mémoire longue), définies dans src/hfrnn/task_map.py.

Modèles (500 neurones, 3 seeds, 1 500 pas, protocole des parties 5 et 6) : régression linéaire sur une fenêtre de
valeurs passées (fenêtre et régularisation choisies sur validation), réservoir classique, RNN, GRU bien initialisé,
LRU, réservoir accordé, réservoir accordé linéaire (type LRU).

    python experiments/29_task_map.py                    # GPU (~1 h 30 sur un T4)
    python experiments/29_task_map.py --quick            # vérification sur CPU (N = 20, 20 pas, 3 tâches)
    python experiments/29_task_map.py --report
Sorties : results/data/task_map_n{N}.json (+ _summary), results/figures/task_map_n{N}.png
"""
import argparse
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn import core, task_map

LR = core.LR
LINEAR = "régression linéaire (fenêtre)"
MODELS = [
    ("réservoir classique", {"kind": "reservoir"}, "gray"),
    ("RNN, lr global × 0.1", {"kind": "convex", "alpha": 1.0, "lr": 0.1 * LR}, "tab:red"),
    ("GRU, init soignée", {"kind": "gru", "rec_init": "ortho"}, "tab:orange"),
    ("LRU", {"kind": "lru", "lr": 5e-4}, "tab:pink"),
    ("réservoir accordé", {"kind": "spectrum"}, "tab:blue"),
    ("réservoir accordé linéaire (type LRU)", {"kind": "linear_tuned"}, "tab:green"),
]
# les portes multiplicatives du GRU sont-elles ce qui manque au réservoir accordé sur Lorenz, Hénon et Wiener-Hammerstein ?
GATED = [
    ("réservoir accordé à portes", {"kind": "gated_tuned", "cell": "gru"}, "tab:cyan"),
    ("accordé sélectif", {"kind": "tuned", "selective": True}, "lightsteelblue"),
]
MODELS = MODELS + GATED
EXISTING = ["RNN, lr global × 0.1", "GRU, init soignée", "LRU"]
OURS = "réservoir accordé"
GATE = "réservoir accordé à portes"


def run(a, path, res):
    import torch
    from hfrnn import torch_models as TM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = list(task_map.TASKS)[:3] if a.quick else list(task_map.TASKS)
    print(f"appareil : {dev} | N = {a.n} | {a.seeds} seeds | {a.steps} pas | {len(names)} tâches", flush=True)
    t0, done = time.time(), 0
    total = len(names) * a.seeds * len(MODELS)
    cache = {}
    for name in names:
        u, y, nt = task_map.load(name)
        R = res["results"].setdefault(name, {})
        if LINEAR not in R:
            e, info = task_map.linear_baseline(u, y, nt)
            R[LINEAR] = {"nmse": e, **info}
        cache[name] = (u, y, nt, 1 if u.ndim == 1 else u.shape[1])
    # d'abord les modèles du premier lot sur toutes les tâches, puis les cellules à portes
    for group in (MODELS[:-len(GATED)], GATED):
        for name in names:
            u, y, nt, d = cache[name]
            R = res["results"][name]
            for seed in range(a.seeds):
                Win, W = TM.make_reservoir(a.n, n_in=d, rho=core.RHO, in_scale=core.IN_SCALE / np.sqrt(d),
                                           sparsity=core.SPARSITY, seed=seed)
                for label, kw, _ in group:
                    cell = R.setdefault(label, {})
                    if str(seed) in cell:
                        continue
                    score, info = TM.train_regression(u, y, Win, W, seed, steps=a.steps, device=dev, n_train=nt,
                                                      **dict(kw))
                    cell[str(seed)] = {"nmse": score, "n_params": info["n_params"]}
                    done += 1
                    el = time.time() - t0
                    left = total - sum(len(res["results"].get(t, {}).get(m, {})) for t in names for m, _, _ in MODELS)
                    print(f"[{el / 60:6.1f} min, reste ≈ {el / done * left / 60:4.0f} min] {name:34s} seed {seed} "
                          f"{label:38s} NMSE = {score:.3g}", flush=True)
                    with open(path, "w") as f:
                        json.dump(res, f, indent=1, ensure_ascii=False)
    with open(path, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


def verdict(diffs):
    """Écart apparié log10 (accordé − autre) : « gagne » s'il est sous 0 de plus de 2 erreurs types et d'au moins
    0.05 (12 %), « perd » dans l'autre sens, « égalité » sinon."""
    m = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs))) if len(diffs) > 1 else 0.0
    if m < -max(2 * se, 0.05):
        return "gagne", m, se
    if m > max(2 * se, 0.05):
        return "perd", m, se
    return "égalité", m, se


def report(res, n):
    R = res["results"]
    names = [t for t in task_map.TASKS if t in R]
    seeds = None
    rows, out = [], {}
    for t in names:
        have = [m for m, _, _ in MODELS[:-len(GATED)] if m in R[t] and R[t][m]]
        k = min(len(R[t][m]) for m in have) if have else 0
        if OURS not in have or k == 0:
            continue
        seeds = k if seeds is None else min(seeds, k)
        ss = [str(s) for s in range(k)]
        v = {m: np.array([R[t][m][s]["nmse"] for s in ss]) for m, _, _ in MODELS
             if m in R[t] and all(s in R[t][m] for s in ss)}
        gm = {m: float(np.exp(np.mean(np.log(x)))) for m, x in v.items()}
        gm[LINEAR] = R[t][LINEAR]["nmse"]
        ex = [m for m in EXISTING if m in v]
        best_ex = min(ex, key=lambda m: gm[m])
        vs_ex = verdict(np.log10(v[OURS]) - np.log10(v[best_ex]))
        vs_lin = verdict(np.log10(v[OURS]) - np.log10(max(gm[LINEAR], 1e-30)))
        _, fam = task_map.TASKS[t]
        out[t] = {"famille": fam, "nmse": gm, "meilleur existant": best_ex, "seeds": k,
                  "contre le meilleur existant": {"résultat": vs_ex[0], "écart log10": vs_ex[1], "erreur type": vs_ex[2]},
                  "contre la régression linéaire": {"résultat": vs_lin[0], "écart log10": vs_lin[1]}}
        if GATE in v:                                         # le réservoir accordé à portes, s'il est calculé
            g = verdict(np.log10(v[GATE]) - np.log10(v[best_ex]))
            gain = verdict(np.log10(v[GATE]) - np.log10(v[OURS]))
            out[t]["portes contre le meilleur existant"] = {"résultat": g[0], "écart log10": g[1], "erreur type": g[2]}
            out[t]["portes contre le réservoir accordé"] = {"écart log10": gain[1], "erreur type": gain[2]}
        rows.append(t)
    if not rows:
        print("Pas encore de résultats.")
        return
    wins = sum(out[t]["contre le meilleur existant"]["résultat"] == "gagne" for t in rows)
    wins_l = sum(out[t]["contre la régression linéaire"]["résultat"] == "gagne" for t in rows)
    print(f"\nCarte des tâches, N = {n}, {seeds} seeds. Réservoir accordé contre le meilleur modèle existant (GRU, LRU, "
          f"RNN) et contre la régression linéaire")
    for t in rows:
        o = out[t]
        a, b = o["contre le meilleur existant"], o["contre la régression linéaire"]
        print(f"  {t:36s} accordé {o['nmse'][OURS]:9.3g} | {o['meilleur existant']:22s} {o['nmse'][o['meilleur existant']]:9.3g}"
              f" : {a['résultat']:8s} ({a['écart log10']:+.2f} ± {a['erreur type']:.2f}) | "
              f"linéaire {o['nmse'][LINEAR]:9.3g} : {b['résultat']}")
    print(f"  victoires de l'accordé : {wins}/{len(rows)} contre les modèles existants, {wins_l}/{len(rows)} contre le linéaire")
    gr = [t for t in rows if "portes contre le meilleur existant" in out[t]]
    wins_g = sum(out[t]["portes contre le meilleur existant"]["résultat"] == "gagne" for t in gr)
    if gr:
        print(f"\n  Réservoir accordé à portes contre le meilleur modèle existant, et gain sur le "
              f"réservoir accordé (log10, négatif = les portes aident)")
        for t in gr:
            o, q = out[t]["portes contre le meilleur existant"], out[t]["portes contre le réservoir accordé"]
            print(f"  {t:36s} portes {out[t]['nmse'][GATE]:9.3g} : {o['résultat']:8s} "
                  f"({o['écart log10']:+.2f} ± {o['erreur type']:.2f}) | contre l'accordé {q['écart log10']:+.2f} ± "
                  f"{q['erreur type']:.2f}")
        print(f"  victoires des portes : {wins_g}/{len(gr)}")
    summary = {"tâches": out, "victoires": {"accordé contre le meilleur existant": [wins, len(rows)],
                                            "accordé contre la régression linéaire": [wins_l, len(rows)],
                                            "portes contre le meilleur existant": [wins_g, len(gr)]}}
    with open(core.out_path("data", f"task_map_n{n}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    cols = [LINEAR] + [m for m, _, _ in MODELS]
    M = np.full((len(rows), len(cols)), np.nan)
    for i, t in enumerate(rows):
        best = min(v for v in out[t]["nmse"].values())
        for j, m in enumerate(cols):
            if m in out[t]["nmse"]:
                M[i, j] = np.log10(max(out[t]["nmse"][m], 1e-30) / max(best, 1e-30))
    fig, ax = plt.subplots(figsize=(17, 0.55 * len(rows) + 3))
    im = ax.imshow(np.clip(M, 0, 3), cmap="RdYlGn_r", vmin=0, vmax=3, aspect="auto")
    for i, t in enumerate(rows):
        for j, m in enumerate(cols):
            if np.isfinite(M[i, j]):
                v = out[t]["nmse"][m]
                ax.text(j, i, f"{v:.2g}", ha="center", va="center", fontsize=7.5,
                        fontweight="bold" if M[i, j] == 0 else "normal")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=25, ha="right", fontsize=8)
    lab = []
    for t in rows:
        a = out[t]["contre le meilleur existant"]
        lab.append(f"{t}  [{out[t]['famille']}]  accordé : {a['résultat']}")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(lab, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, label="log10(NMSE / meilleure NMSE de la ligne), coupé à 3")
    ax.set_title(f"Partie 8.3 — carte des tâches ({n} neurones, {seeds} seeds). Chiffres : NMSE de test ; vert = "
                 f"meilleur de la ligne. À gauche : réservoir accordé contre le meilleur modèle existant "
                 f"({wins}/{len(rows)} victoires)", fontsize=10)
    fig.tight_layout()
    fig.savefig(core.out_path("figures", f"task_map_n{n}.png"), dpi=110)
    print(f"\nFigure : results/figures/task_map_n{n}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=core.STEPS)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n, a.seeds, a.steps = 20, 2, 20
    path = core.out_path("data", f"task_map_n{a.n}{'_quick' if a.quick else ''}.json")
    res = {"args": {"n": a.n, "steps": a.steps}, "results": {}}
    if path.exists():
        old = json.load(open(path))
        if old["args"] == res["args"]:
            res = old
    if not a.report:
        run(a, path, res)
    report(res, a.n)


if __name__ == "__main__":
    main()
