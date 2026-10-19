"""
Partie 8 — Synthèse : tous les résultats déjà calculés, sans rien recalculer.

Pour chaque tâche, une figure : un panneau par expérience (partie de l'étude), barres horizontales triées de la
meilleure (en haut) à la moins bonne, même échelle pour tous les panneaux d'une tâche. Chaque panneau garde son propre
protocole (taille, durée d'entraînement, nombre de seeds), écrit dans son titre : on compare les modèles à l'intérieur
d'un panneau, pas d'un panneau à l'autre.

En plus :
  - erreur contre nombre de paramètres appris, à protocole identique (500 neurones, 1 500 pas, 3 seeds) ;
  - au-delà des tâches synthétiques : MNIST (ligne par ligne, avec recherche, pixel par pixel) et séries réelles ;
  - un tableau du meilleur modèle par tâche et par expérience (results/data/synthesis.json).

Les expériences encore en cours (fichiers absents ou incomplets) sont simplement omises : relancer ce script après
chaque nouveau lot de résultats met la synthèse à jour.

    python experiments/28_synthesis.py      (quelques secondes, CPU)
Sorties : results/figures/synthesis_*.png, results/data/synthesis.json
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn import core

TASKS = ["Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)"]
SLUG = {"Mackey-Glass (+15)": "mackey_glass", "NARMA10": "narma10", "Mémoire (délai 15)": "memoire"}
D = core.RESULTS_DIR / "data"

# couleur par famille (premier motif trouvé dans l'étiquette)
FAMILY = [("non entraîné", "silver"), ("réservoir classique", "gray"), ("rapide", "tab:green"),
          ("directions de Fourier", "tab:green"), ("portes seules", "lightsteelblue"),
          ("porte d'entrée", "darkcyan"), ("à portes", "tab:cyan"), ("linéaire", "tab:green"),
          ("sélectif", "lightsteelblue"), ("accordé", "tab:blue"), ("sculpté", "tab:purple"), ("élagué", "plum"),
          ("repondéré", "plum"), ("tourné", "tab:pink"), ("rang", "tab:olive"), ("retouché", "darkkhaki"),
          ("freiné", "tab:brown"), ("mélange", "teal"), ("ralenti", "salmon"), ("GRU", "tab:orange"),
          ("LSTM", "goldenrod"), ("LRU", "hotpink"), ("RNN", "tab:red")]


def color(label):
    for key, c in FAMILY:
        if key in label:
            return c
    return "black"


def load(name):
    p = D / name
    return json.load(open(p)) if p.exists() else None


def from_table(rows):
    """Tableaux déjà résumés (parties 1 à 3, 100 neurones) : {étiquette: (moyenne, erreur standard)}."""
    return {r["label"]: (r["mean"], r.get("se", 0.0)) for r in rows}


def from_seeds(cell, min_seeds=3):
    """{étiquette: {seed: {"nmse": …}} ou [valeurs]} → {étiquette: (moyenne, erreur standard)}, seeds complètes."""
    out = {}
    for label, v in cell.items():
        vals = list(v) if isinstance(v, list) else [x["nmse"] if isinstance(x, dict) else x for x in v.values()]
        vals = [x for x in vals if x is not None and np.isfinite(x)]
        if len(vals) >= min_seeds:
            vals = np.array(vals)
            out[label] = (float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals))))
    return out


def panels(task):
    """Liste (titre, protocole, {étiquette: (moyenne, se)}) pour une tâche, dans l'ordre de l'étude."""
    P = []
    r = load("comparison.json")
    if r and task in r:
        P.append(("1 · Mélanger", "100 neurones, 1 500 pas, 8 seeds (5 mémoire)", from_table(r[task])))
    r = load("supermask.json")
    if r and task in r:
        P.append(("2 · Sculpté", "100 neurones, 1 500 pas, 8 seeds (5 mémoire)", from_table(r[task]["table"])))
    r = load("rotation.json")
    if r and task in r["comparaison"]:
        P.append(("3-4 · Tourné ou accordé", "100 neurones, 1 500 pas", from_table(r["comparaison"][task])))
    r = load("spectrum_controls_n500.json")
    if r and task in r["results"]:
        P.append(("4.2 · Contrôles du spectre de l'accordé", "500 neurones, 1 500 pas, 3 seeds", from_seeds(r["results"][task])))
    r = load("gpu_regression_n500.json")
    if r and task in r["results"]:
        P.append(("1-4 · Toutes les familles à 500 neurones, combinaisons", "500 neurones, 1 500 pas, 3 seeds",
                  from_seeds(r["results"][task])))
    r = load("improvements_n500.json")
    if r and task in r["results"]:
        m = from_seeds(r["results"][task])
        if any(k for k in m if k.startswith(("accordé ", "contrôle"))):
            P.append(("4.4 · Variantes de l'accordé", "500 neurones, 1 500 pas, 3 seeds", m))
    r = load("gated_n500.json")
    if r and task in r["results"]:
        P.append(("4.5-4.6 · Portes et récurrence linéaire", "500 neurones, 1 500 pas, 3 seeds", from_seeds(r["results"][task])))
    r = load("comparisons_n500.json")
    if r and task in r["results"]:
        P.append(("8.2 · Face aux modèles existants", "500 neurones, 4 000 pas, 3 seeds", from_seeds(r["results"][task])))
    r = load("fair_search_n500.json")
    if r and task in r.get("final", {}):
        m = from_seeds({k: v["seeds"] for k, v in r["final"][task].items()}, min_seeds=5)
        if m:
            P.append(("8.2 · Même budget de recherche", "500 neurones, 1 500 pas, 5 seeds, choix sur validation", m))
    r, f = load("scaling.json"), load("fast_scaling.json")
    big = {}
    if r and "2000" in r["results"] and task in r["results"]["2000"]:
        big |= {f"{k} (N = 2 000)": v for k, v in from_seeds(r["results"]["2000"][task]).items()}
    if f and "16000" in f["results"] and task in f["results"]["16000"]:
        big |= {f"{k} (N = 16 000)": v for k, v in from_seeds(f["results"]["16000"][task]).items()}
    if big:
        P.append(("4.7 · Grandir", "2 000 et 16 000 neurones, 1 500 pas, 3 seeds", big))
    n_mt = 500 if load("multitask_n500.json") else 100      # la version à 500 neurones quand elle existe
    r = load(f"multitask_n{n_mt}.json")
    if r:
        seeds = sorted(r["results"])
        conds = ["accordé, tout par tâche", "accordé, entrées partagées", "RNN, une copie par tâche", "RNN séquentiel",
                 "accordé séquentiel, un réglage"]
        m = from_seeds({c: [r["results"][s][c][task]["final"] for s in seeds] for c in conds})
        if m:
            P.append(("4.8 · Un réservoir, plusieurs tâches",
                      f"{n_mt} neurones, 1 500 pas, {len(seeds)} seeds, toutes tâches apprises", m))
    return P


def task_figure(task, P):
    lo = min(v[0] for _, _, m in P for v in m.values())
    hi = max(v[0] for _, _, m in P for v in m.values())
    ncol = 3
    nrow = int(np.ceil(len(P) / ncol))
    heights = [max(len(m) for _, _, m in P[i * ncol:(i + 1) * ncol]) for i in range(nrow)]
    rows_in = [0.2 * h + 1.0 for h in heights]                # hauteur utile de chaque rangée (pouces)
    H = sum(rows_in) + 0.9 * nrow + 0.9
    fig = plt.figure(figsize=(7.2 * ncol, H))
    gs = fig.add_gridspec(nrow, ncol, height_ratios=rows_in, hspace=0.9 * nrow / sum(rows_in) * nrow / nrow + 0.25,
                          wspace=0.95, left=0.17, right=0.985, top=1 - 1.05 / H, bottom=0.45 / H)
    best = {}
    for i, (title, proto, m) in enumerate(P):
        ax = fig.add_subplot(gs[i // ncol, i % ncol])
        items = sorted(m.items(), key=lambda kv: kv[1][0])
        labs = [k for k, _ in items]
        y = np.arange(len(items))
        ax.barh(y, [v[0] for _, v in items], xerr=[v[1] for _, v in items], color=[color(k) for k in labs],
                capsize=1.5, error_kw={"lw": 0.6})
        ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=7); ax.invert_yaxis()
        ax.get_yticklabels()[0].set_fontweight("bold")
        ax.set_xscale("log"); ax.set_xlim(lo / 2, hi * 2); ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.3, which="major")
        ax.set_title(f"{title}\n{proto}", fontsize=8.5, loc="left")
        best[title] = {"protocole": proto, "meilleur": labs[0], "nmse": items[0][1][0],
                       "deuxième": labs[1] if len(labs) > 1 else None}
    fig.suptitle(f"Synthèse — {task} : NMSE de test (plus bas = mieux, échelle log commune), meilleur modèle en gras",
                 fontsize=13, y=1 - 0.25 / H)
    fig.savefig(core.out_path("figures", f"synthesis_{SLUG[task]}.png"), dpi=100)
    plt.close(fig)
    return best


def n_params_regression():
    """Paramètres appris des modèles de la partie 4.1 (le fichier ne les garde pas) : recalculés sans entraînement."""
    import torch
    from hfrnn import torch_models as TM
    spec = importlib.util.spec_from_file_location("g", Path(__file__).with_name("05_gpu_regression.py"))
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    out = {}
    for label, kw in g.MODELS:
        kw = dict(kw); sp = kw.pop("sparsity", core.SPARSITY)
        for k in ("lr", "lr_mult", "lr_mult_in", "mult", "lr_theta"):
            kw.pop(k, None) if k in ("lr", "lr_theta") else None
        Win, W = TM.make_reservoir(500, rho=core.RHO, in_scale=core.IN_SCALE, sparsity=sp, seed=0)
        kind = kw.pop("kind")
        try:
            out[label] = TM.n_trainable(TM.build_model(Win, W, kind, 0, torch.float32, "cpu", **kw))
        except TypeError:
            pass
    return out


def pareto():
    """Erreur contre paramètres appris, protocole commun : 500 neurones, 1 500 pas, 3 seeds."""
    pts = {t: {} for t in TASKS}
    npar = {}
    r = load("gpu_regression_n500.json")
    if r:
        npar |= n_params_regression()
        for t in TASKS:
            for k, v in from_seeds(r["results"].get(t, {})).items():
                if k in npar:
                    pts[t][k] = (npar[k], v[0])
    for name in ("spectrum_controls_n500.json", "gated_n500.json", "improvements_n500.json"):
        r = load(name)
        if not r:
            continue
        for t in TASKS:
            cell = r["results"].get(t, {})
            for k, v in from_seeds(cell).items():
                p = next(iter(cell[k].values()))["n_params"]
                pts[t].setdefault(k, (p, v[0]))
    if not any(pts.values()):
        return
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.2))
    key = ("réservoir accordé", "réservoir corrigé de rang 4", "réservoir sculpté intégral, dense", "RNN, lr global × 0.1",
           "GRU, init soignée, lr 2e-3", "LSTM, init soignée, lr 5e-4", "réservoir accordé à porte d'entrée",
           "accordé sélectif", "réservoir accordé linéaire (type LRU)", "accordé, 10 directions sur 500")
    for ax, t in zip(axes, TASKS):
        items = [(k, p, e) for k, (p, e) in pts[t].items() if p > 0]
        for k, p, e in items:
            ax.scatter(p, e, s=45 if k in key else 18, color=color(k), edgecolors="black" if k in key else "none",
                       lw=0.6, zorder=3 if k in key else 2)
            if k in key:
                ax.annotate(k.replace(", init soignée", "").replace(", dense", ""), (p, e), fontsize=7,
                            xytext=(4, 3), textcoords="offset points")
        # front de Pareto : aucun modèle n'a à la fois moins de paramètres et moins d'erreur
        front, best = [], np.inf
        for k, p, e in sorted(items, key=lambda x: (x[1], x[2])):
            if e < best:
                front.append((p, e)); best = e
        if front:
            ax.step(*zip(*front), where="post", color="black", lw=0.8, alpha=0.5)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.grid(alpha=0.3, which="both")
        ax.set_xlabel("paramètres appris hors lecture (log)"); ax.set_ylabel("NMSE test (log)"); ax.set_title(t)
    fig.suptitle("Synthèse — erreur contre nombre de paramètres appris (500 neurones, 1 500 pas, 3 seeds). "
                 "Trait noir : front de Pareto.", fontsize=12)
    fig.tight_layout(); fig.savefig(core.out_path("figures", "synthesis_pareto.png"), dpi=105)
    plt.close(fig)


def beyond():
    """MNIST (erreur %) et séries réelles (NMSE relative à la prévision naïve)."""
    panels_ = []
    r = load("gpu_smnist_n256.json")
    if r:
        m = {k: (100 * (1 - np.mean(v)), 100 * np.std(v, ddof=1) / np.sqrt(len(v))) for k, v in r["accuracy"].items()}
        panels_.append(("MNIST ligne par ligne, réglages communs", "256 neurones, 3 époques, 3 seeds", m, "erreur de test (%)"))
    r = load("mnist_search_summary.json")
    if r:
        last = max(r, key=int)
        m = {f"{k}, {w}": (v[w]["moyenne"], v[w]["se"]) for k, v in r[last].items() for w in ("défaut", "recherche")}
        panels_.append(("MNIST ligne par ligne, même budget de recherche", f"256 neurones, {last} époques, 3 seeds",
                        m, "erreur de test (%)"))
    r = load("pixel_mnist_summary.json")
    if r:
        m = {k: (v["erreur test (%)"], v["se"]) for k, v in r.items()}
        panels_.append(("MNIST pixel par pixel", "128 neurones, 784 pas, 3 époques", m, "erreur de test (%)"))
    r = load("nonlinear_benchmarks_summary.json")
    if r:
        for k, v in r.items():
            m = {name: (v["rmse"][name], v["erreur type"][name]) for name in v["rmse"]}
            panels_.append((f"Système réel · {k}", f"500 neurones, simulation", m, f"RMSE de test ({v['unité']})"))
    r = load("uea_n128_summary.json")
    if r and r.get("précision moyenne"):
        m = {k: (100 * (1 - v), 0.0) for k, v in r["précision moyenne"].items()}
        panels_.append(("Archive UEA, 10 jeux", "128 neurones, 3 seeds, erreur moyenne sur les jeux", m,
                        "erreur de test moyenne (%)"))
    r = load("real_data.json")
    if r:
        for t, v in r.items():
            m = from_table(v["table"]) | {"prévision naïve (variation nulle)": (v["persistance"], 0.0)}
            panels_.append((f"Série réelle · {t}", "100 neurones, 1 500 pas, 5 seeds", m, "NMSE test"))
    if not panels_:
        return
    ncol = 3
    nrow = int(np.ceil(len(panels_) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2 * ncol, 5.2 * nrow), squeeze=False)
    for ax in axes.flat[len(panels_):]:
        ax.axis("off")
    for ax, (title, proto, m, xl) in zip(axes.flat, panels_):
        items = sorted(m.items(), key=lambda kv: kv[1][0])
        y = np.arange(len(items))
        ax.barh(y, [v[0] for _, v in items], xerr=[v[1] for _, v in items],
                color=["black" if "naïve" in k else color(k) for k, _ in items], capsize=1.5, error_kw={"lw": 0.6})
        ax.set_yticks(y); ax.set_yticklabels([k for k, _ in items], fontsize=7); ax.invert_yaxis()
        ax.set_xscale("log"); ax.set_xlabel(xl, fontsize=8); ax.set_title(f"{title}\n{proto}", fontsize=9, loc="left")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Synthèse — au-delà des tâches synthétiques (barre plus courte = mieux)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.975)); fig.savefig(core.out_path("figures", "synthesis_beyond.png"), dpi=100)
    plt.close(fig)


def main():
    out = {}
    for t in TASKS:
        P = panels(t)
        if P:
            out[t] = task_figure(t, P)
            print(f"\n{t} — meilleur modèle par expérience :")
            for title, b in out[t].items():
                print(f"  {title:52s} {b['meilleur']:45s} {b['nmse']:.3g}   ({b['protocole']})")
    pareto()
    beyond()
    with open(core.out_path("data", "synthesis.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print("\nFigures : results/figures/synthesis_{mackey_glass,narma10,memoire,pareto,beyond}.png")


if __name__ == "__main__":
    main()
