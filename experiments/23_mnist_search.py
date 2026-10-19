"""
Partie 8.3 — MNIST séquentiel avec recherche d'hyperparamètres équitable.

Dans la partie 8.1, tous les modèles recevaient les mêmes réglages (learning rate 1e-3, rayon spectral 0.9, échelle
d'entrée 0.1, densité 10 %), ceux du RNN de référence : notre modèle n'avait aucun réglage propre. Ici, chaque modèle
reçoit **le même budget de recherche** :

    réservoir accordé   lr, lr de l'amplitude g, lr des entrées, rayon spectral, échelle d'entrée, densité
    RNN                 lr, lr de la partie récurrente, lr des entrées, rayon spectral, échelle d'entrée, densité
    GRU                 lr, initialisation de la matrice récurrente (par défaut ou orthogonale)

Protocole :
  1. recherche aléatoire (même nombre d'essais par modèle, mêmes graines de tirage) : 3 époques, 1 seed, 50 000 images
     d'entraînement, score = précision sur 10 000 images de **validation** (jamais sur le test) ;
  2. le meilleur essai de chaque modèle, et la configuration par défaut de la partie 8.1, sont ré-entraînés sur
     plusieurs seeds, 10 époques ; la précision **test** est mesurée une seule fois par époque, sans jamais servir à
     choisir quoi que ce soit ;
  3. (option) les mêmes réglages à plus grande taille d'état, sans nouvelle recherche.

Les 10 000 images de test officielles ne servent qu'à l'étape 2 et 3.

    python experiments/23_mnist_search.py                     # GPU : N = 256, 24 essais, 3 seeds (~1 h 15)
    python experiments/23_mnist_search.py --quick --synthetic  # vérification sur CPU (données synthétiques)
    python experiments/23_mnist_search.py --report
Sorties : results/data/mnist_search.json, results/figures/mnist_search.png
"""
import argparse
import json
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn import core
from hfrnn import torch_models as TM

# ----------------------------------------------------------------------------------------------------------
# Modèles, espaces de recherche, configurations par défaut (celles de la partie 8.1)
# ----------------------------------------------------------------------------------------------------------
RES = {"rho": ("uni", 0.6, 1.4), "in_scale": ("log", 0.02, 0.5), "density": ("choice", [0.05, 0.1, 0.3, 1.0])}
MODELS = {
    "réservoir accordé": {
        "kind": "spectrum", "color": "tab:blue",
        "space": {"lr": ("log", 3e-4, 1e-2), "mult": ("log", 0.3, 10.0), "lr_mult_in": ("log", 0.3, 3.0), **RES},
        "default": {"lr": 1e-3, "mult": 1.0, "lr_mult_in": 1.0, "rho": 0.9, "in_scale": 0.1, "density": 0.1}},
    "RNN": {
        "kind": "convex", "color": "tab:red",
        "space": {"lr": ("log", 3e-4, 1e-2), "lr_mult": ("log", 0.03, 1.0), "lr_mult_in": ("log", 0.3, 3.0), **RES},
        "default": {"lr": 1e-3, "lr_mult": 1.0, "lr_mult_in": 1.0, "rho": 0.9, "in_scale": 0.1, "density": 0.1}},
    "GRU": {
        "kind": "gru", "color": "tab:orange",
        "space": {"lr": ("log", 3e-4, 1e-2), "rec_init": ("choice", ["default", "ortho"])},
        "default": {"lr": 1e-3, "rec_init": "default"}},
}
RESERVOIR_KEYS = ("rho", "in_scale", "density")


def sample(model, i, base_seed=2026):
    """i-ème configuration tirée au hasard pour `model` : déterministe, donc une recherche interrompue reprend à
    l'identique, et chaque modèle reçoit un tirage indépendant du même nombre d'essais."""
    rng = np.random.default_rng([base_seed, list(MODELS).index(model), i])
    cfg = {}
    for k, spec in MODELS[model]["space"].items():
        if spec[0] == "log":
            cfg[k] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        elif spec[0] == "uni":
            cfg[k] = float(rng.uniform(spec[1], spec[2]))
        else:
            cfg[k] = spec[1][int(rng.integers(len(spec[1])))]
    return cfg


def fit(data, model, cfg, seed, epochs, n, test, device, batch):
    kw = {k: v for k, v in cfg.items() if k not in RESERVOIR_KEYS and k != "lr"}
    kind = MODELS[model]["kind"]
    if kind == "convex":
        kw["alpha"] = 1.0
    return TM.fit_classifier(data, kind, seed, n=n, epochs=epochs, lr=cfg["lr"], batch=batch, device=device, test=test,
                             **{k: cfg[k] for k in RESERVOIR_KEYS if k in cfg}, **kw)


# ----------------------------------------------------------------------------------------------------------
# Données
# ----------------------------------------------------------------------------------------------------------
def synthetic_data(n_train, n_val, n_test, seed=0, device="cpu"):
    """Dix gabarits 28 × 28 bruités : un jeu jouet qui se classe vite, pour vérifier le pipeline sans MNIST."""
    rng = np.random.default_rng(seed)
    tpl = rng.random((10, 28, 28)) ** 2

    def make(n):
        y = rng.integers(0, 10, n)
        x = np.clip(0.9 * tpl[y] + 0.3 * rng.standard_normal((n, 28, 28)), 0, 1)
        return torch.tensor(x, dtype=torch.float32, device=device), torch.tensor(y, device=device)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = make(n_train), make(n_val), make(n_test)
    return dict(Xtr=Xtr, ytr=ytr, Xva=Xva, yva=yva, Xte=Xte, yte=yte)


def mnist_data(n_val, device, limit=None):
    """MNIST : 60 000 images d'entraînement dont n_val tirées (graine fixe) pour la validation ; 10 000 de test."""
    from torchvision import datasets
    tr = datasets.MNIST("data_mnist", train=True, download=True)
    te = datasets.MNIST("data_mnist", train=False, download=True)
    X, y = tr.data.float().div(255), tr.targets
    perm = torch.randperm(len(y), generator=torch.Generator().manual_seed(0))
    va, trn = perm[:n_val], perm[n_val:]
    Xte, yte = te.data.float().div(255), te.targets
    if limit:
        trn, Xte, yte = trn[:limit], Xte[:limit], yte[:limit]
    d = dict(Xtr=X[trn], ytr=y[trn], Xva=X[va], yva=y[va], Xte=Xte, yte=yte)
    return {k: v.to(device) for k, v in d.items()}


# ----------------------------------------------------------------------------------------------------------
# Expérience
# ----------------------------------------------------------------------------------------------------------
def load(path, args, keys):
    """Résultats existants si les réglages qui définissent l'étape sont identiques, sinon nouveau départ."""
    if path.exists():
        old = json.load(open(path))
        if all(old["args"].get(k) == args.get(k) for k in keys):
            return old
    return {"args": dict(args), "search": {}, "best": {}, "final": {}}


def save(path, res):
    with open(path, "w") as f:
        json.dump(res, f, indent=1, ensure_ascii=False)


def best_trial(search):
    """Meilleur essai (hors configuration par défaut) : précision de validation à la dernière époque."""
    trials = {k: v for k, v in search.items() if k != "default"}
    return max(trials, key=lambda k: (trials[k]["val"][-1], -int(k)))


def run(a, path):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.synthetic:
        data = synthetic_data(a.train, a.val, a.test, device=dev)
    else:
        data = mnist_data(a.val, dev, limit=a.train if a.quick else None)
    args = vars(a).copy()
    res = load(path, args, ("n", "trials", "search_epochs", "val", "batch", "quick", "synthetic"))
    res["args"] = args
    print(f"appareil : {dev} | N = {a.n} | {a.trials} essais × {len(MODELS)} modèles, {a.search_epochs} époques | "
          f"finale : {a.seeds} seeds, {a.final_epochs} époques, tailles {a.sizes}", flush=True)
    t0 = time.time()

    # 1. recherche : essai par essai pour tous les modèles (un arrêt précoce laisse des budgets égaux)
    for i in range(a.trials):
        for m in MODELS:
            S = res["search"].setdefault(m, {})
            if str(i) in S:
                continue
            cfg = sample(m, i)
            r = fit(data, m, cfg, 0, a.search_epochs, a.n, False, dev, a.batch)
            S[str(i)] = {"cfg": cfg, "val": r["val"], "n_params": r["n_params"], "diverged": r["diverged"]}
            print(f"[{(time.time() - t0) / 60:6.1f} min] recherche {m:18s} essai {i:2d} validation {r['val'][-1]:.4f}"
                  f"{' (divergé)' if r['diverged'] else ''}  lr = {cfg['lr']:.2g}", flush=True)
            save(path, res)
    for m in MODELS:                                           # configuration par défaut de la partie 8.1
        S = res["search"].setdefault(m, {})
        if "default" not in S:
            r = fit(data, m, MODELS[m]["default"], 0, a.search_epochs, a.n, False, dev, a.batch)
            S["default"] = {"cfg": MODELS[m]["default"], "val": r["val"], "n_params": r["n_params"],
                            "diverged": r["diverged"]}
            print(f"[{(time.time() - t0) / 60:6.1f} min] défaut    {m:18s} validation {r['val'][-1]:.4f}", flush=True)
            save(path, res)
        res["best"][m] = best_trial(S)

    # 2 et 3. configuration retenue (et configuration par défaut à la taille de la recherche), plusieurs seeds
    for n in a.sizes:
        F = res["final"].setdefault(str(n), {})
        for m in MODELS:
            for which in ("défaut", "recherche") if n == a.n else ("recherche",):
                cfg = (MODELS[m]["default"] if which == "défaut" else res["search"][m][res["best"][m]]["cfg"])
                cell = F.setdefault(m, {}).setdefault(which, {})
                for seed in range(a.seeds):
                    if str(seed) in cell and len(cell[str(seed)]["test"]) == a.final_epochs:
                        continue
                    r = fit(data, m, cfg, seed, a.final_epochs, n, True, dev, a.batch)
                    cell[str(seed)] = {"val": r["val"], "test": r["test"], "n_params": r["n_params"],
                                       "diverged": r["diverged"], "seconds": r["seconds"]}
                    print(f"[{(time.time() - t0) / 60:6.1f} min] final N = {n:5d} {m:18s} {which:9s} seed {seed} "
                          f"test {r['test'][-1]:.4f} (époque 3 : {r['test'][min(2, len(r['test']) - 1)]:.4f})",
                          flush=True)
                    save(path, res)
    return res


# ----------------------------------------------------------------------------------------------------------
# Rapport
# ----------------------------------------------------------------------------------------------------------
def final_errors(res, n, m, which, epoch):
    cell = res["final"][str(n)][m][which]
    return np.array([1 - cell[s]["test"][min(epoch, len(cell[s]["test"])) - 1] for s in sorted(cell)])


def report(res):
    a = res["args"]
    tag = "_quick" if a.get("quick") else ""
    n0 = str(a["n"])
    print(f"\nRecherche d'hyperparamètres : {a['trials']} essais par modèle, {a['search_epochs']} époques, "
          f"validation sur {a['val']} images (aucun essai ne voit le test)")
    print(f"  {'modèle':18s} {'défaut':>8s} {'meilleur':>9s} {'médiane':>8s} {'pire':>8s}   erreur de validation (%), meilleure configuration")
    for m in MODELS:
        S = res["search"][m]
        errs = np.array([100 * (1 - v["val"][-1]) for k, v in S.items() if k != "default"])
        b = S[res["best"][m]]["cfg"]
        print(f"  {m:18s} {100 * (1 - S['default']['val'][-1]):8.2f} {errs.min():9.2f} {np.median(errs):8.2f} "
              f"{errs.max():8.2f}   " + ", ".join(f"{k} = {v:.3g}" if isinstance(v, float) else f"{k} = {v}"
                                                   for k, v in b.items()))
    out = {}
    for epoch in (3, a["final_epochs"]):
        print(f"\nErreur de test (%), N = {n0}, {a['seeds']} seeds, après {epoch} époques")
        for m in MODELS:
            row = {w: final_errors(res, n0, m, w, epoch) * 100 for w in ("défaut", "recherche")}
            out.setdefault(str(epoch), {})[m] = {w: {"moyenne": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v)))
                                                      if len(v) > 1 else 0.0} for w, v in row.items()}
            print(f"  {m:18s} défaut {row['défaut'].mean():6.2f}   après recherche {row['recherche'].mean():6.2f} "
                  f"± {out[str(epoch)][m]['recherche']['se']:.2f}   n_params = "
                  f"{res['final'][n0][m]['recherche']['0']['n_params']}")
        mine, rnn, gru = ("réservoir accordé", "RNN", "GRU")
        for other in (rnn, gru):
            d = final_errors(res, n0, other, "recherche", epoch) * 100 - final_errors(res, n0, mine, "recherche", epoch) * 100
            print(f"    écart apparié, après recherche : {other} − accordé = {d.mean():+.2f} points "
                  f"(± {d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0:.2f}), positif = accordé meilleur")
    for n in res["final"]:
        if n != n0:
            print(f"\nErreur de test (%), N = {n}, mêmes réglages que N = {n0}, après {a['final_epochs']} époques")
            for m in MODELS:
                e = final_errors(res, n, m, "recherche", a["final_epochs"]) * 100
                print(f"  {m:18s} {e.mean():6.2f} ± {e.std(ddof=1) / np.sqrt(len(e)) if len(e) > 1 else 0:.2f}   "
                      f"n_params = {res['final'][n][m]['recherche']['0']['n_params']}")
    with open(core.out_path("data", f"mnist_search_summary{tag}.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))
    rng = np.random.default_rng(0)
    ax = axes[0]
    for j, m in enumerate(MODELS):
        S = res["search"][m]
        e = [100 * (1 - v["val"][-1]) for k, v in S.items() if k != "default"]
        ax.scatter(j + 0.18 * rng.standard_normal(len(e)), e, s=22, alpha=0.6, color=MODELS[m]["color"])
        ax.scatter([j], [100 * (1 - S["default"]["val"][-1])], marker="_", s=900, color="black", zorder=3,
                   label="configuration par défaut (8.1)" if j == 0 else None)
        ax.scatter([j], [min(e)], marker="*", s=230, color=MODELS[m]["color"], edgecolors="black", zorder=4,
                   label="meilleur essai" if j == 0 else None)
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(list(MODELS)); ax.set_yscale("log")
    ax.set_ylabel("erreur de validation (%, log)"); ax.legend(fontsize=8, loc="best")
    ax.set_title(f"{a['trials']} essais par modèle : plus la dispersion est grande, plus le réglage compte")
    ax = axes[1]
    for m in MODELS:
        S = res["search"][m]
        ks = [k for k in S if k != "default"]
        ax.scatter([S[k]["cfg"]["lr"] for k in ks], [100 * (1 - S[k]["val"][-1]) for k in ks], s=26, alpha=0.7,
                   color=MODELS[m]["color"], label=m)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("learning rate"); ax.set_ylabel("erreur de validation (%)")
    ax.legend(fontsize=8); ax.set_title("sensibilité au learning rate")
    ax = axes[2]
    sizes = sorted(res["final"], key=int)
    groups = [(n, w) for n in sizes for w in ("défaut", "recherche") if w in res["final"][n][next(iter(MODELS))]]
    wbar = 0.8 / len(groups)
    for g, (n, w) in enumerate(groups):
        e = [final_errors(res, n, m, w, a["final_epochs"]) * 100 for m in MODELS]
        ax.bar(np.arange(len(MODELS)) + (g - (len(groups) - 1) / 2) * wbar, [v.mean() for v in e], wbar,
               yerr=[v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0 for v in e], capsize=3,
               color=[MODELS[m]["color"] for m in MODELS], alpha=0.45 if w == "défaut" else 1.0,
               hatch="//" if n != n0 else None, edgecolor="black", linewidth=0.6,
               label=f"N = {n}, {'réglages par défaut' if w == 'défaut' else 'après recherche'}")
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(list(MODELS)); ax.set_yscale("log")
    ax.set_ylabel(f"erreur de test (%, log), {a['final_epochs']} époques"); ax.legend(fontsize=7)
    ax.set_title("résultat final sur le test (une seule mesure par époque)")
    fig.suptitle("Partie 8.2 — MNIST séquentiel avec le même budget de recherche pour tous les modèles",
                 fontsize=12)
    fig.tight_layout(); fig.savefig(core.out_path("figures", f"mnist_search{tag}.png"), dpi=115)
    print(f"\nFigure : results/figures/mnist_search{tag}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256, help="taille d'état pendant la recherche")
    ap.add_argument("--trials", type=int, default=24, help="essais par modèle")
    ap.add_argument("--search-epochs", dest="search_epochs", type=int, default=3)
    ap.add_argument("--final-epochs", dest="final_epochs", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sizes", type=int, nargs="*", default=None, help="tailles d'état de l'étape finale (défaut : N 1024)")
    ap.add_argument("--val", type=int, default=10000)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--quick", action="store_true", help="vérification rapide")
    ap.add_argument("--synthetic", action="store_true", help="données synthétiques (pas de téléchargement de MNIST)")
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=10000)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n, a.trials, a.search_epochs, a.final_epochs, a.seeds = 32, 2, 1, 3, 2
        a.val, a.train, a.test, a.sizes = 500, 1500, 500, a.sizes or [32, 48]
    if a.sizes is None:
        a.sizes = [a.n, 1024]
    if a.n not in a.sizes:
        a.sizes = [a.n] + a.sizes
    tag = "_quick" if a.quick else ""
    path = core.out_path("data", f"mnist_search{tag}.json")
    if a.report:
        if not path.exists():
            raise SystemExit(f"Pas de résultats à résumer : {path}")
        res = load(path, vars(a), ())
    else:
        res = run(a, path)
    report(res)


if __name__ == "__main__":
    main()
