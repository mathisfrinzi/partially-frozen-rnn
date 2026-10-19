"""
Partie 8.3 — Identification de systèmes réels : les benchmarks de nonlinearbenchmark.org.

Des mesures sur de vrais systèmes physiques, avec un découpage entraînement/test officiel et des résultats publiés :

    Silverbox                  circuit électronique qui imite un oscillateur de Duffing (Wigren et Schoukens, 2013)
    Wiener-Hammerstein (2009)  filtre, diode, filtre : circuit électronique (Schoukens et al., 2009)
    Cascaded Tanks             deux réservoirs d'eau en cascade, avec débordement (Schoukens et al., 2016) ; 1 024
                               points d'entraînement seulement

Tâche : la **simulation**. Le modèle ne reçoit que l'entrée u(t) et doit sortir y(t), sans jamais voir les sorties
mesurées pendant le test. Chaque modèle est entraîné sur la partie entraînement (fenêtres de 60 pas, 1 500 pas
d'Adam, lecture ridge finale) ; le test part d'un état nul et les premiers pas (fenêtre d'initialisation officielle)
ne sont pas comptés. Erreur : RMSE dans l'unité du système (comparable aux tableaux publiés sur le site) et NMSE.

Données téléchargées par le paquet `nonlinear-benchmarks` (pip install nonlinear-benchmarks), sur Colab.
    python experiments/30_nonlinear_benchmarks.py              # GPU (~45 min sur un T4, estimation)
    python experiments/30_nonlinear_benchmarks.py --synthetic --quick   # vérification sur CPU, sans téléchargement
    python experiments/30_nonlinear_benchmarks.py --report
Sorties : results/data/nonlinear_benchmarks.json (+ _summary), results/figures/nonlinear_benchmarks.png
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn import core, task_map

_spec = importlib.util.spec_from_file_location("task_map_exp", Path(__file__).with_name("29_task_map.py"))
TMX = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(TMX)
MODELS, LINEAR, EXISTING, OURS = TMX.MODELS, TMX.LINEAR, TMX.EXISTING, TMX.OURS
# --extra : readout carré, relais + accordé, réservoir nilpotent (parties 6 et 7).
EXTRA = [("réservoir accordé, readout carré", {"kind": "spectrum", "square": True}, "tab:purple"),
         ("relais aléatoire + RNN accordé, readout carré",
          {"kind": "coupled", "form": "relay_rand", "a_block": "tuned", "square": True}, "tab:brown"),
         ("Jordan-Chevalley + partie nilpotente", {"kind": "decomp", "form": "jordan_SN"}, "tab:olive")]
# --extra2 : réservoir à biais et entrées appris (sans g) et réservoir cisaillé, plein et en bande (parties 4 et 5).
EXTRA2 = [("réservoir à biais et entrées appris (sans g)", {"kind": "spectrum", "learn_g": False}, "tab:cyan"),
          ("réservoir cisaillé en bande, k = 3", {"kind": "decomp", "form": "iwasawa_N", "band": 3}, "tab:pink"),
          ("réservoir cisaillé", {"kind": "decomp", "form": "iwasawa_N"}, "tab:gray")]


# Silverbox : le paquet nonlinear-benchmarks (1.0.1) donne le même nom « test SB multisine » à deux tests différents
# (le multisinus et la flèche complète) ; on nomme donc les tests par leur position, dans l'ordre du paquet.
TEST_NAMES = {"Silverbox": ["multisinus", "flèche (avec extrapolation)", "flèche (sans extrapolation)"]}


def test_name(bench, t, i):
    names = TEST_NAMES.get(bench)
    return names[i] if names and i < len(names) else (t.name or f"test {i}")


def load_benchmarks(synthetic=False, quick=False):
    """{nom: {"train": (u, y), "tests": {nom: (u, y, init)}, "unit": str, "scale": float}} ; un benchmark qui ne se
    télécharge pas est remplacé par {"erreur": message}."""
    out = {}
    if synthetic:
        for i, name in enumerate(["Silverbox (simulé)", "Cascaded Tanks (simulé)"]):
            n = 600 if quick else 6000
            u, y, _ = task_map.wiener_hammerstein(T=2 * n, seed=30 + i)
            out[name] = {"train": (u[:n], y[:n]), "tests": {"test": (u[n:], y[n:], 50)}, "unit": "u.a.", "scale": 1.0}
        return out
    import nonlinear_benchmarks as nb

    def arr(d):
        return np.asarray(d.u, float).ravel(), np.asarray(d.y, float).ravel()

    loaders = {
        "Silverbox": (lambda: nb.Silverbox(), "mV", 1000.0),
        "Wiener-Hammerstein (2009)": (lambda: nb.WienerHammerBenchMark(), "mV", 1000.0),
        "Cascaded Tanks": (lambda: nb.Cascaded_Tanks(), "V", 1.0),
    }
    for name, (load, unit, scale) in loaders.items():
        try:
            train, tests = load()
            tests = tests if isinstance(tests, tuple) else (tests,)
            out[name] = {"train": arr(train), "unit": unit, "scale": scale,
                         "tests": {test_name(name, t, i): (*arr(t), int(t.state_initialization_window_length or 50))
                                   for i, t in enumerate(tests)}}
        except Exception as e:                                       # site indisponible : on continue sans lui
            out[name] = {"erreur": f"{type(e).__name__}: {e}"}
            print(f"{name} : téléchargement impossible ({out[name]['erreur']})", flush=True)
    return out


def evaluate(model, ut, yt, tests, ystd, dev):
    """Lecture ridge sur les états d'entraînement (départ d'un état nul), puis simulation de chaque test depuis un
    état nul ; renvoie {test: {"nmse": .., "rmse": .. (unité normalisée × ystd)}}."""
    import torch
    from hfrnn.torch_models import ridge, as_batch
    with torch.no_grad():
        X = model(as_batch(ut))[0]
        w = ridge(X[core.WASHOUT:], yt[core.WASHOUT:])
        res = {}
        for name, (u, y, init) in tests.items():
            U = torch.as_tensor(u, dtype=ut.dtype, device=dev)
            Xs = model(as_batch(U))[0].double()
            p = (torch.cat([Xs, torch.ones(len(Xs), 1, dtype=Xs.dtype, device=Xs.device)], 1) @ w).cpu().numpy()
            e = p[init:] - y[init:]
            res[name] = {"nmse": float(np.mean(e ** 2) / y[init:].var()), "rmse": float(np.sqrt(np.mean(e ** 2)) * ystd)}
    return res


def run(a, path, res):
    import torch
    from hfrnn import torch_models as TM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = load_benchmarks(a.synthetic, a.quick)
    print(f"appareil : {dev} | N = {a.n} | {a.seeds} seeds | {a.steps} pas", flush=True)
    t0 = time.time()
    for name, B in data.items():
        R = res["results"].setdefault(name, {})
        if "erreur" in B:
            R["erreur"] = B["erreur"]
            continue
        R.pop("erreur", None)
        u, y = B["train"]
        mu, su, my, sy = u.mean(), u.std(), y.mean(), y.std()
        un, yn = (u - mu) / su, (y - my) / sy
        tests = {k: ((tu - mu) / su, (ty - my) / sy, init) for k, (tu, ty, init) in B["tests"].items()}
        R["infos"] = {"unité": B["unit"], "échelle": B["scale"], "points d'entraînement": len(u),
                      "tests": {k: len(v[0]) for k, v in tests.items()}}
        ystd = sy * B["scale"]
        if LINEAR not in R:                                          # fenêtre linéaire (FIR), choisie sur validation
            R[LINEAR] = {}
            for k, (tu, ty, init) in tests.items():
                uu, yy = np.concatenate([un, tu]), np.concatenate([yn, ty])
                e, info = task_map.linear_baseline(uu, yy, len(un), lags=(10, 20, 50, 100))
                R[LINEAR][k] = {"nmse": e, "rmse": float(np.sqrt(e * ty.var()) * ystd), **info}
        # entraînement : l'entrée de test est concaténée seulement pour que train_regression ait un bloc « test »
        # (son score est ignoré) ; les fenêtres d'apprentissage ne sont tirées que dans la partie entraînement
        first = next(iter(tests.values()))
        uu = np.concatenate([un, first[0][:max(200, len(first[0]) // 10)]])
        yy = np.concatenate([yn, first[1][:len(uu) - len(un)]])
        for seed in range(a.seeds):
            Win, W = TM.make_reservoir(a.n, n_in=1, rho=core.RHO, in_scale=core.IN_SCALE, sparsity=core.SPARSITY,
                                       seed=seed)
            for label, kw, _ in MODELS:
                cell = R.setdefault(label, {})
                if str(seed) in cell:
                    continue
                t1 = time.time()
                try:
                    _, info = TM.train_regression(uu, yy, Win, W, seed, steps=a.steps, device=dev, n_train=len(un),
                                                  return_model=True, **dict(kw))
                    m = info["model"]
                    ut = torch.as_tensor(un, dtype=torch.float32, device=dev)
                    yt = torch.as_tensor(yn, dtype=torch.float32, device=dev)
                    t2 = time.time()
                    cell[str(seed)] = {"tests": evaluate(m, ut, yt, tests, ystd, dev), "n_params": info["n_params"],
                                       "secondes": [round(t2 - t1, 1), round(time.time() - t2, 1)]}
                    R.get("erreurs", {}).pop(f"{label}, seed {seed}", None)   # erreur d'un essai précédent réparée
                    if "erreurs" in R and not R["erreurs"]:
                        del R["erreurs"]
                except Exception as e:                               # on note l'erreur et on passe au suivant
                    import traceback
                    traceback.print_exc()
                    R.setdefault("erreurs", {})[f"{label}, seed {seed}"] = f"{type(e).__name__}: {e}"[:500]
                    print(f"[{(time.time() - t0) / 60:6.1f} min] {name:28s} seed {seed} {label:38s} ERREUR {e}",
                          flush=True)
                    if dev == "cuda":
                        torch.cuda.empty_cache()
                    continue
                finally:
                    with open(path, "w") as f:
                        json.dump(res, f, indent=1, ensure_ascii=False)
                main = next(iter(cell[str(seed)]["tests"].values()))
                sec = cell[str(seed)]["secondes"]
                print(f"[{(time.time() - t0) / 60:6.1f} min] {name:28s} seed {seed} {label:38s} "
                      f"RMSE {main['rmse']:.4g} {B['unit']}  NMSE {main['nmse']:.3g}  "
                      f"(entraînement {sec[0]:.0f} s, évaluation {sec[1]:.0f} s)", flush=True)
                with open(path, "w") as f:
                    json.dump(res, f, indent=1, ensure_ascii=False)
        with open(path, "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)


def report(res, tag=""):
    R = res["results"]
    out = {}
    print(f"\nIdentification de systèmes réels (simulation), N = {res['args']['n']}. RMSE de test (moyenne sur les "
          f"seeds) ; meilleur en premier")
    panels = []
    for name, B in R.items():
        if "erreur" in B:
            print(f"  {name} : non téléchargé ({B['erreur'][:80]})")
            continue
        for k, e in B.get("erreurs", {}).items():
            print(f"  {name}, {k} : erreur ({e[:120]})")
        unit = B["infos"]["unité"]
        for test in B[LINEAR]:
            rows = {LINEAR: (B[LINEAR][test]["rmse"], 0.0, B[LINEAR][test]["nmse"])}
            for label, _, _ in MODELS:
                if label in B and B[label]:
                    r = np.array([v["tests"][test]["rmse"] for v in B[label].values()])
                    nm = np.array([v["tests"][test]["nmse"] for v in B[label].values()])
                    rows[label] = (float(r.mean()), float(r.std(ddof=1) / np.sqrt(len(r))) if len(r) > 1 else 0.0,
                                   float(np.exp(np.log(nm).mean())))
            key = f"{name} — {test}"
            out[key] = {"unité": unit, "rmse": {k: v[0] for k, v in rows.items()},
                        "erreur type": {k: v[1] for k, v in rows.items()}, "nmse": {k: v[2] for k, v in rows.items()}}
            print(f"  {key} ({unit})")
            for k, (r, se, nm) in sorted(rows.items(), key=lambda kv: kv[1][0]):
                print(f"    {k:40s} RMSE {r:9.4g} ± {se:.2g}   NMSE {nm:.3g}")
            panels.append((key, unit, rows))
    with open(core.out_path("data", f"nonlinear_benchmarks{tag}_summary.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    if not panels:
        return
    colors = {LINEAR: "black", **{m: c for m, _, c in MODELS}}
    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 2.6 * len(panels) + 1), squeeze=False)
    for ax, (key, unit, rows) in zip(axes[:, 0], panels):
        items = sorted(rows.items(), key=lambda kv: kv[1][0])
        ax.barh(range(len(items)), [v[0] for _, v in items], xerr=[v[1] for _, v in items],
                color=[colors.get(k, "gray") for k, _ in items])
        ax.set_yticks(range(len(items))); ax.set_yticklabels([k for k, _ in items], fontsize=8)
        ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlabel(f"RMSE de test ({unit}, log)")
        ax.set_title(key, fontsize=10)
    fig.suptitle("Partie 8.6 — identification de systèmes réels, en simulation (nonlinearbenchmark.org)", fontsize=11)
    fig.tight_layout()
    fig.savefig(core.out_path("figures", f"nonlinear_benchmarks{tag}.png"), dpi=110)
    print(f"\nFigure : results/figures/nonlinear_benchmarks{tag}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=core.STEPS)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--extra", action="store_true", help="ajoute les modèles des parties 2.8 et 3 (readout carré, relais)")
    ap.add_argument("--extra2", action="store_true", help="ajoute sans g et le réservoir cisaillé (plein et en bande)")
    a = ap.parse_args()
    global MODELS
    if a.extra:
        MODELS = MODELS + EXTRA
    if a.extra2:
        MODELS = MODELS + EXTRA2
    if a.quick:
        a.n, a.seeds, a.steps = 20, 2, 20
    tag = "_quick" if a.quick else ""
    path = core.out_path("data", f"nonlinear_benchmarks{tag}.json")
    res = {"args": {"n": a.n, "steps": a.steps, "synthetic": a.synthetic}, "results": {}}
    if path.exists():
        old = json.load(open(path))
        if old["args"] == res["args"]:
            res = old
    if not a.report:
        run(a, path, res)
    report(res, tag)


if __name__ == "__main__":
    main()
