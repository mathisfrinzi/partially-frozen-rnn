"""
Partie 8.3 : pourquoi le réservoir accordé bat-il le GRU de 75× sur l'amplificateur de puissance ?

Le premier contrôle (33_amplifier_control.py) a écarté un mauvais réglage : avec deux learning rates et 4 000 pas,
le GRU reste vers 0.2 et le réservoir accordé descend à 0.0016. Restent deux explications, que ce script sépare :

  1. Le surapprentissage. 3 000 points d'entraînement pour 750 000 paramètres (GRU) ou 251 000 (RNN), contre 1 500
     pour le réservoir accordé. Si c'est la cause, l'erreur d'entraînement du GRU est bien plus basse que son erreur
     de test, et l'écart se referme avec plus de données. On mesure les deux erreurs, avec 3 000 et 20 000 points.
  2. Une référence trop faible. La comparaison juste, dans ce domaine, est le polynôme à mémoire généralisé (GMP),
     le modèle standard de prédistorsion des amplificateurs. On l'ajoute (ridge, ordres et mémoire choisis sur
     validation).
  3. Une représentation défavorable au GRU. Dans la littérature (OpenDPD, RNN à normalisation de phase), les réseaux
     récurrents reçoivent des entrées enrichies (amplitude, puissances de l'amplitude) en plus de I et Q, et
     marchent bien. Notre GRU, qui reçoit I et Q seuls, échoue : on lui donne (et au réservoir accordé) les entrées
     [I, Q, |x|, |x|²] pour voir si l'écart disparaît.

    python experiments/34_amplifier_deep_dive.py            # GPU (~45 min)
    python experiments/34_amplifier_deep_dive.py --quick    # vérification rapide
    python experiments/34_amplifier_deep_dive.py --report
Sorties : results/data/amplifier_deep_dive.json (+ _summary), results/figures/amplifier_deep_dive.png
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
CONFIGS = [
    ("réservoir accordé", {"kind": "spectrum"}, "tab:blue"),
    ("GRU, init soignée, lr 2e-3", {"kind": "gru", "rec_init": "ortho"}, "tab:orange"),
    ("RNN, lr standard", {"kind": "convex", "alpha": 1.0}, "salmon"),
    ("réservoir accordé à portes", {"kind": "gated_tuned", "cell": "gru"}, "tab:cyan"),
    ("réservoir accordé, entrées enrichies", {"kind": "spectrum", "_feat": True}, "navy"),
    ("GRU, entrées enrichies", {"kind": "gru", "rec_init": "ortho", "_feat": True}, "darkorange"),
]
SIZES = [3000, 20000]
STEPS = [1500, 4000]
GMP = "polynôme à mémoire généralisé (GMP)"
LINEAR = "régression linéaire (fenêtre)"


def enrich(u, nt):
    """Entrées enrichies, comme dans la littérature des amplificateurs : [I, Q, |x|, |x|²], normalisées sur l'entraînement."""
    a = np.hypot(u[:, 0], u[:, 1])
    F = np.stack([u[:, 0], u[:, 1], a, a ** 2], 1)
    return task_map._norm(F, np.zeros(len(F)), nt)[0]


def load(n_train, quick=False):
    """Une seule longue série ; la petite taille en est la fin (même partie test pour toutes les tailles)."""
    big = 2000 if quick else max(SIZES)
    u, y, _ = task_map.power_amplifier(T=big + task_map.T_TEST)
    u, y = u[-(n_train + task_map.T_TEST):], y[-(n_train + task_map.T_TEST):]
    return task_map._norm(u, y, n_train)


def train_nmse(model, u, y, nt, dev):
    """Erreur d'entraînement : lecture ridge apprise sur l'entraînement, évaluée sur l'entraînement."""
    import torch
    from hfrnn.torch_models import ridge, as_batch
    with torch.no_grad():
        ut = torch.as_tensor(u, dtype=torch.float32, device=dev)
        yt = torch.as_tensor(y, dtype=torch.float32, device=dev).double()
        X = model(as_batch(ut))[0].double()
        w = ridge(X[core.WASHOUT:nt], yt[core.WASHOUT:nt])
        p = torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype, device=X.device)], 1) @ w
        e = p[core.WASHOUT:nt] - yt[core.WASHOUT:nt]
        return float((e ** 2).mean() / yt[core.WASHOUT:nt].var(unbiased=False))


def run(a, path, res):
    import torch
    from hfrnn import torch_models as TM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"appareil : {dev} | N = {a.n} | {a.seeds} seeds | tailles {a.sizes} | pas {a.steps}", flush=True)
    t0 = time.time()
    for size in a.sizes:
        u, y, nt = load(size, a.quick)
        R = res["results"].setdefault(str(size), {})
        if GMP not in R:
            e, info = task_map.memory_polynomial_baseline(u, y, nt)
            R[GMP] = {"test": e, **info}
            e, info = task_map.linear_baseline(u, y, nt)
            R[LINEAR] = {"test": e, **info}
            print(f"[{(time.time() - t0) / 60:6.1f} min] {size} points : GMP {R[GMP]['test']:.3g} "
                  f"({ {k: v for k, v in R[GMP].items() if k != 'test'} }), linéaire {R[LINEAR]['test']:.3g}",
                  flush=True)
        for seed in range(a.seeds):
            for label, kw, _ in CONFIGS:
                cell = R.setdefault(label, {})
                if str(seed) in cell:
                    continue
                kw = dict(kw)
                uu = enrich(u, nt) if kw.pop("_feat", False) else u
                d = uu.shape[1]
                Win, W = TM.make_reservoir(a.n, n_in=d, rho=core.RHO, in_scale=core.IN_SCALE / np.sqrt(d),
                                           sparsity=core.SPARSITY, seed=seed)
                _, info = TM.train_regression(uu, y, Win, W, seed, steps=max(a.steps), device=dev, n_train=nt,
                                              eval_at=tuple(a.steps), return_model=True, **kw)
                cell[str(seed)] = {"test": {str(k): float(v) for k, v in info["checkpoints"].items()},
                                   "entraînement": train_nmse(info["model"], uu, y, nt, dev),
                                   "paramètres": info["n_params"]}
                c = cell[str(seed)]
                print(f"[{(time.time() - t0) / 60:6.1f} min] {size} points, seed {seed} {label:36s} "
                      + "  ".join(f"test {k} pas : {v:.3g}" for k, v in c["test"].items())
                      + f"  entraînement : {c['entraînement']:.3g}", flush=True)
                with open(path, "w") as f:
                    json.dump(res, f, indent=1, ensure_ascii=False)
                del info
                if dev == "cuda":
                    torch.cuda.empty_cache()
        with open(path, "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)


def gmean(xs):
    return float(np.exp(np.mean(np.log(xs))))


def report(res, tag=""):
    sizes = [str(s) for s in res["args"]["sizes"]]
    last = str(max(res["args"]["steps"]))
    out = {}
    print(f"\nAmplificateur de puissance, N = {res['args']['n']} : NMSE (moyenne géométrique sur les seeds), "
          f"{last} pas")
    for s in sizes:
        R = res["results"].get(s, {})
        if not R:
            continue
        row = {}
        print(f"  {s} points d'entraînement")
        for k in (GMP, LINEAR):
            if k in R:
                row[k] = {"test": R[k]["test"]}
                print(f"    {k:40s} test {R[k]['test']:.3g}")
        for label, _, _ in CONFIGS:
            if label not in R or not R[label]:
                continue
            v = list(R[label].values())
            row[label] = {"test": gmean([x["test"][last] for x in v]), "entraînement": gmean([x["entraînement"]
                                                                                            for x in v]),
                          "paramètres": v[0]["paramètres"], "seeds": len(v)}
            r = row[label]
            print(f"    {label:40s} test {r['test']:.3g}   entraînement {r['entraînement']:.3g}   "
                  f"(test / entraînement : {r['test'] / r['entraînement']:.3g} ; {r['paramètres']} paramètres, "
                  f"{r['seeds']} seeds)")
        out[s] = row

    summary = {"tailles": out}
    with open(core.out_path("data", f"amplifier_deep_dive{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    fig, ax = plt.subplots(1, len(out), figsize=(6.5 * len(out), 4.6), squeeze=False)
    for j, (s, row) in enumerate(out.items()):
        A = ax[0, j]
        labels = list(row)
        cols = {GMP: "black", LINEAR: "dimgray", **{l: c for l, _, c in CONFIGS}}
        labels = [l for l in labels if l in cols]
        x = np.arange(len(labels))
        A.bar(x, [row[l]["test"] for l in labels], 0.6, color=[cols[l] for l in labels], label="test")
        tr = [row[l].get("entraînement", np.nan) for l in labels]
        A.scatter(x, tr, marker="_", s=400, color="red", zorder=3, label="entraînement")
        A.set_xticks(x); A.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        A.set_yscale("log"); A.set_title(f"{s} points d'entraînement", fontsize=10)
        A.set_ylabel("NMSE (log)"); A.legend(fontsize=8)
    fig.suptitle(f"Partie 8.5 — amplificateur de puissance : surapprentissage et référence du domaine "
                 f"({res['args']['n']} neurones)", fontsize=11)
    fig.tight_layout()
    fig.savefig(core.out_path("figures", f"amplifier_deep_dive{tag}.png"), dpi=110)
    print(f"\nFigure : results/figures/amplifier_deep_dive{tag}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    a.sizes, a.steps = SIZES, STEPS
    if a.quick:
        a.n, a.seeds, a.sizes, a.steps = 40, 1, [600, 2000], [5, 10]
    tag = "_quick" if a.quick else ""
    path = core.out_path("data", f"amplifier_deep_dive{tag}.json")
    res = {"args": {"n": a.n, "sizes": a.sizes, "steps": a.steps}, "results": {}}
    if path.exists():
        old = json.load(open(path))
        if old["args"] == res["args"]:
            res = old
    if not a.report:
        run(a, path, res)
    report(res, tag)


if __name__ == "__main__":
    main()
