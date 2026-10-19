"""
Partie 8.2 — Face aux modèles existants, avec le même budget de recherche pour tous.

Dans la partie 6.2, chaque modèle existant a eu quatre réglages (choisis sur le test), et nos modèles un seul. Ici,
chaque modèle reçoit le même budget : la configuration par défaut de la partie 6 plus 12 essais tirés au hasard,
jugés sur une **validation** prise à la fin de la partie entraînement de chaque série (les 750 derniers pas des
3 000 d'entraînement). Les 1 000 pas de test ne servent qu'à la fin : la meilleure configuration de chaque modèle
est ré-entraînée sur 5 seeds avec toute la partie entraînement, et mesurée une fois sur le test.

    GRU                                  lr, initialisation (par défaut ou orthogonale)
    LRU                                  lr, une ou deux couches, rayon minimal et maximal des valeurs propres
    réservoir accordé                    lr, lr des amplitudes g, lr des entrées, rayon spectral, échelle d'entrée,
                                         densité
    réservoir accordé à porte d'entrée   lr, rayon spectral, échelle d'entrée, densité

    python experiments/26_fair_search.py                 # N = 500, GPU (~2 h 50 sur un T4)
    python experiments/26_fair_search.py --quick         # vérification sur CPU
    python experiments/26_fair_search.py --report

Pour partager le travail entre plusieurs processus sur un seul GPU, ou pour ajouter des modèles ou des tâches sans toucher
à ce qui est déjà calculé : `--tasks`, `--models` (sous-chaînes de noms) et `--tag` (un fichier par processus). Le rapport
fusionne tous les fichiers fair_search_n{N}*.json. Les tâches Lorenz et Hénon s'ajoutent à la fin de la liste, et les
modèles cisaillés à la fin des modèles, si bien que les tirages d'essais déjà faits restent identiques.
Sorties : results/data/fair_search_n{N}[tag].json, results/figures/fair_search_n{N}.png
"""
import argparse
import glob
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn import core, tasks

TASKS = ["Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)"]
CHAOS = ["Lorenz (+10)", "Hénon (+3)"]
ALL_TASKS = TASKS + CHAOS
RES = {"rho": ("uni", 0.6, 1.4), "in_scale": ("log", 0.1, 2.0), "density": ("choice", [0.05, 0.1, 0.3, 1.0])}
MODELS = {
    "GRU": {"kind": "gru", "color": "tab:orange", "existing": True,
            "space": {"lr": ("log", 3e-4, 1e-2), "rec_init": ("choice", ["default", "ortho"])},
            "default": {"lr": 2e-3, "rec_init": "ortho"}},
    "LRU": {"kind": "lru", "color": "tab:pink", "existing": True,
            "space": {"lr": ("log", 3e-4, 1e-2), "layers": ("choice", [1, 2]), "r_min": ("uni", 0.0, 0.9),
                      "r_max": ("uni", 0.9, 0.999)},
            "default": {"lr": 2e-3, "layers": 1, "r_min": 0.5, "r_max": 0.99}},
    "réservoir accordé": {"kind": "spectrum", "color": "tab:blue", "existing": False,
                          "space": {"lr": ("log", 3e-4, 1e-2), "mult": ("log", 0.3, 10.0),
                                    "lr_mult_in": ("log", 0.3, 3.0), **RES},
                          "default": {"lr": 2e-3, "mult": 1.0, "lr_mult_in": 1.0, "rho": 0.9, "in_scale": 0.5,
                                      "density": 0.1}},
    "réservoir accordé à porte d'entrée": {"kind": "gated_tuned", "color": "darkcyan", "existing": False,
                                           "space": {"lr": ("log", 3e-4, 1e-2), **RES},
                                           "default": {"lr": 2e-3, "rho": 0.9, "in_scale": 0.5, "density": 0.1},
                                           "fixed": {"cell": "mingru"}},
}
# Ces deux modèles cherchent en plus leur nombre d'unités ; ils sont placés en fin de MODELS pour que les tirages des autres
# modèles restent identiques (une recherche déjà faite reprend sans rien recalculer).
SIZES = ("choice", [20, 40, 60, 100, 200, 500])
MODELS["GRU, taille cherchée"] = {"kind": "gru", "color": "tab:red", "existing": True,
                                  "space": {**MODELS["GRU"]["space"], "n": SIZES},
                                  "default": {**MODELS["GRU"]["default"], "n": 500}}
MODELS["LRU, taille cherchée"] = {"kind": "lru", "color": "tab:purple", "existing": True,
                                  "space": {**MODELS["LRU"]["space"], "n": SIZES},
                                  "default": {**MODELS["LRU"]["default"], "n": 500}}
# Le réservoir dont seuls les biais et les entrées apprennent (récurrence = W₀ exactement, « sans g »), avec le même budget.
MODELS["réservoir à biais et entrées appris (sans g)"] = {
    "kind": "spectrum", "color": "tab:cyan", "existing": False,
    "space": {"lr": ("log", 3e-4, 1e-2), "lr_mult_in": ("log", 0.3, 3.0), **RES},
    "default": {"lr": 2e-3, "lr_mult_in": 1.0, "rho": 0.9, "in_scale": 0.5, "density": 0.1},
    "fixed": {"learn_g": False}}
# Les réservoirs cisaillés : le nombre k de sur-diagonales apprises est cherché comme un hyperparamètre, sur validation, comme tout
# le reste (on ne le règle donc pas sur le test, tâche par tâche).
BAND = ("choice", [1, 3, 10, 30, 100])
MODELS["cisaillé en bande (k cherché)"] = {
    "kind": "decomp", "color": "tab:green", "existing": False,
    "space": {"lr": ("log", 3e-4, 1e-2), "band": BAND, **RES},
    "default": {"lr": 2e-3, "band": 10, "rho": 0.9, "in_scale": 0.5, "density": 0.1}, "fixed": {"form": "iwasawa_N"}}
MODELS["cisaillé en bande + g (k cherché)"] = {
    "kind": "decomp", "color": "tab:olive", "existing": False,
    "space": {"lr": ("log", 3e-4, 1e-2), "band": BAND, **RES},
    "default": {"lr": 2e-3, "band": 10, "rho": 0.9, "in_scale": 0.5, "density": 0.1}, "fixed": {"form": "iwasawa_AN"}}
MODELS["cisaillé à porte d'entrée (k cherché)"] = {
    "kind": "shear_gated", "color": "tab:purple", "existing": False,
    "space": {"lr": ("log", 3e-4, 1e-2), "band": ("choice", [3, 10, 30]), **RES},
    "default": {"lr": 2e-3, "band": 10, "rho": 0.9, "in_scale": 0.5, "density": 0.1},
    "fixed": {"cell": "mingru", "maps": ("shear", "shear", "shear")}}
MODELS["cisaillé linéaire à porte d'entrée (k cherché)"] = {
    "kind": "shear_gated", "color": "tab:brown", "existing": False,
    "space": {"lr": ("log", 3e-4, 1e-2), "band": ("choice", [3, 10, 30]), **RES},
    "default": {"lr": 2e-3, "band": 10, "rho": 0.9, "in_scale": 0.5, "density": 0.1},
    "fixed": {"cell": "lin_gate", "maps": ("shear", "shear", "shear")}}
RESERVOIR_KEYS = ("rho", "in_scale", "density")


def sample(model, task, i, base_seed=7):
    """i-ème essai, tiré de façon déterministe pour (modèle, tâche) : une recherche interrompue reprend à l'identique."""
    rng = np.random.default_rng([base_seed, list(MODELS).index(model), ALL_TASKS.index(task), i])
    cfg = {}
    for k, spec in MODELS[model]["space"].items():
        if spec[0] == "log":
            cfg[k] = float(np.exp(rng.uniform(np.log(spec[1]), np.log(spec[2]))))
        elif spec[0] == "uni":
            cfg[k] = float(rng.uniform(spec[1], spec[2]))
        else:
            v = spec[1][int(rng.integers(len(spec[1])))]
            cfg[k] = v.item() if hasattr(v, "item") else v
    return cfg


def fit(u, y, model, cfg, seed, n, steps, device):
    """Entraîne et renvoie (NMSE sur la dernière partie de la série fournie, paramètres appris)."""
    from hfrnn import torch_models as TM
    spec = MODELS[model]
    kw = {k: v for k, v in cfg.items() if k not in RESERVOIR_KEYS and k not in ("lr", "n")} | spec.get("fixed", {})
    n = int(cfg.get("n", n))                                  # taille cherchée (GRU et LRU du bloc 11)
    Win, W = TM.make_reservoir(n, rho=cfg.get("rho", core.RHO), in_scale=cfg.get("in_scale", core.IN_SCALE),
                               sparsity=cfg.get("density", core.SPARSITY), seed=seed)
    try:
        s, info = TM.train_regression(u, y, Win, W, seed, kind=spec["kind"], steps=steps, lr=cfg["lr"], device=device,
                                      **kw)
    except (RuntimeError, ValueError) as e:                 # essai qui diverge (états infinis, ridge singulier) : échec
        print(f"  essai divergent ({type(e).__name__}: {str(e)[:80]}) : compté comme une erreur infinie", flush=True)
        import torch
        m = TM.build_model(Win, W, spec["kind"], seed, torch.float32, "cpu", **kw)
        return float("inf"), TM.n_trainable(m)
    return (s if np.isfinite(s) else float("inf")), info["n_params"]


def load_task(task):
    if task in CHAOS:
        from hfrnn import task_map
        u, y, _ = task_map.load(task)
        return u, y
    return tasks.load(task)


def validation_split(task):
    """La partie entraînement seule (3 000 pas) : train_regression y prend 75 % pour apprendre (2 250 pas) et
    mesure l'erreur sur les 750 suivants, qui servent de validation. Le test (pas 3 000 à 4 000) n'est jamais vu."""
    u, y = load_task(task)
    nt = core.n_train(u)
    return u[:nt], y[:nt]


def best_key(S):
    return min(S, key=lambda k: (S[k]["val"], k != "défaut", k))


def run(a, path, res):
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"appareil : {dev} | N = {a.n} | {a.trials} essais + défaut par modèle et par tâche | {a.steps} pas | "
          f"finale : {a.seeds} seeds", flush=True)
    t0 = time.time()
    # un motif qui est exactement un nom de modèle ne désigne que lui ; sinon, tous les noms qui le contiennent
    names = [m for m in MODELS if not a.models or m in a.models
             or any(w in m for w in a.models if w not in MODELS)]
    for t in a.tasks:
        uv, yv = validation_split(t)
        for i in ["défaut"] + list(range(a.trials)):        # essai par essai pour tous les modèles : budgets égaux
            for m in names:
                S = res["search"].setdefault(t, {}).setdefault(m, {})
                if str(i) in S:
                    continue
                cfg = MODELS[m]["default"] if i == "défaut" else sample(m, t, i)
                v, npar = fit(uv, yv, m, cfg, 0, a.n, a.steps, dev)
                S[str(i)] = {"cfg": cfg, "val": v, "n_params": npar}
                print(f"[{(time.time() - t0) / 60:6.1f} min] {t[:12]:12s} {m:36s} essai {str(i):>6s} "
                      f"validation {v:.3g}", flush=True)
                with open(path, "w") as f:
                    json.dump(res, f, indent=1, ensure_ascii=False)
    for t in a.tasks:
        u, y = load_task(t)
        for m in names:
            S = res["search"][t][m]
            k = best_key(S)
            cell = res["final"].setdefault(t, {}).setdefault(m, {"essai": k, "cfg": S[k]["cfg"], "seeds": {}})
            for seed in range(a.seeds):
                if str(seed) in cell["seeds"]:
                    continue
                s, npar = fit(u, y, m, S[k]["cfg"], seed, a.n, a.steps, dev)
                cell["seeds"][str(seed)] = {"nmse": s, "n_params": npar}
                print(f"[{(time.time() - t0) / 60:6.1f} min] final {t[:12]:12s} {m:36s} (essai {k}) seed {seed} "
                      f"test {s:.3g}", flush=True)
                with open(path, "w") as f:
                    json.dump(res, f, indent=1, ensure_ascii=False)


def load_merged(n, quick=False):
    """Le fichier principal et tous les fichiers à étiquette (un par processus), fusionnés."""
    base = core.out_path("data", f"fair_search_n{n}{'_quick' if quick else ''}.json")
    res = json.load(open(base)) if base.exists() else None
    for f in sorted(glob.glob(str(base).replace(".json", "_*.json"))):
        if f.endswith("_summary.json"):
            continue
        part = json.load(open(f))
        if res is None:
            res = part
            continue
        for k in ("search", "final"):
            for t, D in part[k].items():
                for m, v in D.items():
                    res[k].setdefault(t, {}).setdefault(m, v)
    return res


def report(res):
    a = res["args"]
    summary = {}
    fig, axes = plt.subplots(2, len(res["final"]), figsize=(6.6 * len(res["final"]), 10), squeeze=False)
    for j, t in enumerate(res["final"]):
        F = res["final"][t]
        seeds = min(len(F[m]["seeds"]) for m in F)
        if seeds == 0:
            continue
        L = {m: np.log10([F[m]["seeds"][str(s)]["nmse"] for s in range(seeds)]) for m in F}
        ex = [m for m in F if MODELS[m]["existing"]]
        if not ex:                                          # tâche où aucun modèle existant n'a encore tourné
            continue
        ref = min(ex, key=lambda m: L[m].mean())
        print(f"\n{t} — N = {a['n']}, {a['steps']} pas, choix sur validation, test sur {seeds} seeds. "
              f"Référence : meilleur modèle existant après recherche = {ref}")
        for m in F:
            d = L[m] - L[ref]
            se = d.std(ddof=1) / np.sqrt(seeds) if seeds > 1 else 0.0
            S = res["search"][t][m]
            vals = np.array([v["val"] for v in S.values()])
            summary.setdefault(t, {})[m] = {"nmse": float(10 ** L[m].mean()), "écart à la référence": float(d.mean()),
                                            "se": float(se), "essai retenu": F[m]["essai"], "cfg": F[m]["cfg"],
                                            "n_params": F[m]["seeds"]["0"]["n_params"]}
            print(f"  {m:36s} {F[m]['seeds']['0']['n_params']:>8d} param.  test {10 ** L[m].mean():.3g}  "
                  f"vs réf. {d.mean():+.2f} ± {se:.2f} (×{10 ** -d.mean():.2f})   essai retenu {F[m]['essai']:>6s}  "
                  f"(validation : défaut {S['défaut']['val']:.3g}, meilleur {vals.min():.3g}, médiane {np.median(vals):.3g})")
        ax = axes[0, j]
        ms = list(F)
        ax.bar(range(len(ms)), [10 ** L[m].mean() for m in ms], color=[MODELS[m]["color"] for m in ms],
               yerr=[np.std(10 ** L[m], ddof=1) / np.sqrt(seeds) if seeds > 1 else 0 for m in ms], capsize=3)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=20, ha="right", fontsize=8); ax.set_yscale("log")
        ax.set_ylabel("NMSE test (log)"); ax.set_title(f"{t} : après recherche, {seeds} seeds")
        ax = axes[1, j]
        for i, m in enumerate(ms):
            S = res["search"][t][m]
            v = [x["val"] for k, x in S.items() if k != "défaut" and np.isfinite(x["val"])]
            ax.scatter(i + 0.15 * np.random.default_rng(i).standard_normal(len(v)), v, s=20, alpha=0.6,
                       color=MODELS[m]["color"])
            ax.scatter([i], [S["défaut"]["val"]], marker="_", s=700, color="black", zorder=3,
                       label="configuration de la partie 6" if i == 0 else None)
        ax.set_xticks(range(len(ms))); ax.set_xticklabels(ms, rotation=20, ha="right", fontsize=8); ax.set_yscale("log")
        ax.set_ylabel("NMSE de validation (log)"); ax.set_title("les essais de la recherche"); ax.legend(fontsize=8)
    with open(core.out_path("data", f"fair_search_n{a['n']}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    fig.suptitle("Partie 6.4 — même budget de recherche pour tous, choix sur validation, test mesuré une fois", fontsize=12)
    fig.tight_layout(); fig.savefig(core.out_path("figures", f"fair_search_n{a['n']}.png"), dpi=110)
    print(f"\nFigure : results/figures/fair_search_n{a['n']}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=core.STEPS)
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--models", nargs="*", help="sous-chaînes des noms de modèles à entraîner (défaut : tous)")
    ap.add_argument("--tag", default="", help="suffixe du fichier de résultats (un fichier par processus)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n, a.trials, a.seeds, a.steps = 24, 2, 2, 10
    tag = f"n{a.n}" + ("_quick" if a.quick else "") + a.tag
    path = core.out_path("data", f"fair_search_{tag}.json")
    res = {"args": {"n": a.n, "steps": a.steps, "trials": a.trials}, "search": {}, "final": {}}
    if path.exists():
        old = json.load(open(path))
        if old["args"] == res["args"]:
            res = old
    if not a.report:
        run(a, path, res)
    merged = load_merged(a.n, a.quick)
    if merged:
        report(merged)


if __name__ == "__main__":
    main()
