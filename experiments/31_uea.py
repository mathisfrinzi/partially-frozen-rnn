"""
Partie 8.3 — Classer des séries temporelles réelles : l'archive UEA (séries multivariées).

Dix jeux de données publics et standard (Bagnall et al., 2018), courts et de faible dimension, sur lesquels les
réservoirs ont déjà été évalués (Bianchi et al., 2020) : voix, gestes, écriture, mouvements, capteurs portés.

    JapaneseVowels (12 coefficients, voix)          SpokenArabicDigits (13, voix)
    CharacterTrajectories (3, écriture)             UWaveGestureLibrary (3, accéléromètre)
    ArticularyWordRecognition (9, capteurs articulatoires)   Epilepsy (3, accéléromètre au poignet)
    NATOPS (24, gestes)                             BasicMotions (6, montre connectée)
    Libras (2, langue des signes)                   RacketSports (6, montre connectée)

Protocole : 128 neurones ; 20 % du jeu d'entraînement officiel mis de côté pour la validation (stratifié) ;
chaque modèle choisit son learning rate (1e-3 ou 3e-3) sur la validation (seed 0), puis 3 seeds ; précision mesurée
une seule fois sur le jeu de test officiel, à la fin de l'entraînement (environ 600 pas d'Adam). Les séries de
longueurs différentes sont complétées par des zéros au début. Classifieur : dernier état et état moyen, puis une
couche linéaire.

    python experiments/31_uea.py                 # GPU, télécharge les données avec aeon (~1 h 15 sur un T4, estimation)
    python experiments/31_uea.py --quick --synthetic   # vérification sur CPU
    python experiments/31_uea.py --report
Sorties : results/data/uea_n{N}.json (+ _summary), results/figures/uea_n{N}.png
"""
import argparse
import json
import math
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn import core

DATASETS = ["JapaneseVowels", "SpokenArabicDigits", "CharacterTrajectories", "UWaveGestureLibrary",
            "ArticularyWordRecognition", "Epilepsy", "NATOPS", "BasicMotions", "Libras", "RacketSports"]
MODELS = [
    ("réservoir classique", {"kind": "reservoir"}, "gray"),
    ("RNN", {"kind": "convex", "alpha": 1.0}, "tab:red"),
    ("GRU, init soignée", {"kind": "gru", "rec_init": "ortho"}, "tab:orange"),
    ("LRU", {"kind": "lru"}, "tab:pink"),
    ("réservoir accordé", {"kind": "spectrum"}, "tab:blue"),
    ("réservoir accordé linéaire (type LRU)", {"kind": "linear_tuned"}, "tab:green"),
]
LRS = [1e-3, 3e-3]


def pad(X):
    """Liste de séries (d, L_i) ou tableau (n, d, L) -> tableau (n, L_max, d), zéros au début."""
    X = [np.asarray(x, float) for x in X]
    L = max(x.shape[1] for x in X)
    out = np.zeros((len(X), L, X[0].shape[0]))
    for i, x in enumerate(X):
        out[i, L - x.shape[1]:] = x.T
    return out


def split_val(y, frac=0.2, seed=0):
    """Indices entraînement / validation, stratifiés par classe (au moins un exemple de validation par classe)."""
    r = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(y):
        idx = r.permutation(np.where(y == c)[0])
        k = max(1, int(round(frac * len(idx)))) if len(idx) > 1 else 0
        va += list(idx[:k]); tr += list(idx[k:])
    return np.array(sorted(tr)), np.array(sorted(va))


def synthetic(name, quick):
    """Données factices de même forme (classes = fréquences différentes), pour tester sans téléchargement."""
    r = np.random.default_rng(len(name))
    n, L, d, k = (60, 30, 3, 3) if quick else (300, 60, 6, 4)
    def make(m):
        y = r.integers(0, k, m)
        t = np.arange(L)
        X = np.sin(0.2 * (y[:, None, None] + 1) * t[None, None, :] + r.uniform(0, 6, (m, d, 1)))
        return X + 0.3 * r.normal(size=X.shape), y
    return make(n), make(n // 2)


def load(name, synthetic_data=False, quick=False):
    if synthetic_data:
        (Xtr, ytr), (Xte, yte) = synthetic(name, quick)
    else:
        from aeon.datasets import load_classification
        Xtr, ytr = load_classification(name, split="train")
        Xte, yte = load_classification(name, split="test")
    classes = sorted(set(np.asarray(ytr).tolist()))
    to_int = {c: i for i, c in enumerate(classes)}
    ytr = np.array([to_int[c] for c in np.asarray(ytr).tolist()])
    yte = np.array([to_int[c] for c in np.asarray(yte).tolist()])
    Xtr, Xte = pad(Xtr), pad(Xte)
    L = max(Xtr.shape[1], Xte.shape[1])                      # même longueur pour tous (zéros au début)
    Xtr = np.concatenate([np.zeros((len(Xtr), L - Xtr.shape[1], Xtr.shape[2])), Xtr], 1)
    Xte = np.concatenate([np.zeros((len(Xte), L - Xte.shape[1], Xte.shape[2])), Xte], 1)
    mu, sd = Xtr.reshape(-1, Xtr.shape[2]).mean(0), Xtr.reshape(-1, Xtr.shape[2]).std(0) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    return Xtr, ytr, Xte, yte, len(classes)


def to_device(Xtr, ytr, Xte, yte, dev):
    import torch
    tr, va = split_val(ytr)
    f = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)   # noqa: E731
    g = lambda a: torch.as_tensor(a, dtype=torch.long, device=dev)      # noqa: E731
    return {"Xtr": f(Xtr[tr]), "ytr": g(ytr[tr]), "Xva": f(Xtr[va]), "yva": g(ytr[va]), "Xte": f(Xte), "yte": g(yte)}


def run(a, path, res):
    import torch
    from hfrnn import torch_models as TM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    names = DATASETS[:2] if a.quick else DATASETS
    print(f"appareil : {dev} | N = {a.n} | {a.seeds} seeds | environ {a.steps} pas d'Adam", flush=True)
    t0 = time.time()
    for name in names:
        R = res["results"].setdefault(name, {})
        try:
            Xtr, ytr, Xte, yte, k = load(name, a.synthetic, a.quick)
        except Exception as e:
            R["erreur"] = f"{type(e).__name__}: {e}"
            print(f"{name} : chargement impossible ({R['erreur'][:120]})", flush=True)
            continue
        R.pop("erreur", None)
        data = to_device(Xtr, ytr, Xte, yte, dev)
        ntr, L, d = data["Xtr"].shape
        batch = max(4, min(32, ntr // 4))
        epochs = max(3, math.ceil(a.steps / (ntr // batch)))
        R["infos"] = {"classes": k, "entraînement": int(ntr), "validation": int(len(data["yva"])),
                      "test": int(len(yte)), "longueur": int(L), "dimensions": int(d), "époques": epochs,
                      "lot": batch}

        def go(label, lr, seed):
            cell = R.setdefault(label, {}).setdefault(f"{lr:g}", {})
            if str(seed) in cell:
                return cell[str(seed)]
            kw = dict(next(m for l, m, _ in MODELS if l == label))
            kind = kw.pop("kind")
            r = TM.fit_classifier(data, kind, seed, n=a.n, epochs=epochs, lr=lr, batch=batch, device=dev, test=True,
                                  rho=core.RHO, in_scale=0.5 / np.sqrt(d), density=core.SPARSITY, **kw)
            cell[str(seed)] = {"val": r["val"][-1], "test": r["test"][-1], "n_params": r["n_params"],
                               "diverged": r["diverged"], "seconds": r["seconds"]}
            print(f"[{(time.time() - t0) / 60:6.1f} min] {name:26s} {label:38s} lr {lr:g} seed {seed}  "
                  f"validation {r['val'][-1]:.3f}  test {r['test'][-1]:.3f}", flush=True)
            with open(path, "w") as f:
                json.dump(res, f, indent=1, ensure_ascii=False)
            return cell[str(seed)]
        for label, _, _ in MODELS:
            for lr in LRS:
                go(label, lr, 0)
            lr = best_lr(R, label)
            for seed in range(1, a.seeds):
                go(label, lr, seed)


def best_lr(R, label):
    return max(LRS, key=lambda lr: R[label][f"{lr:g}"]["0"]["val"])


def report(res, n):
    R = res["results"]
    names = [d for d in DATASETS if d in R and "infos" in R[d]]
    out, acc = {}, {}
    print(f"\nArchive UEA, {n} neurones : précision de test (moyenne sur les seeds, learning rate choisi sur la "
          f"validation)")
    for name in names:
        row = {}
        for label, _, _ in MODELS:
            if label not in R[name] or not all(f"{lr:g}" in R[name][label] and "0" in R[name][label][f"{lr:g}"]
                                               for lr in LRS):
                continue
            runs = R[name][label][f"{best_lr(R[name], label):g}"]
            t = np.array([v["test"] for v in runs.values()])
            row[label] = (float(t.mean()), float(t.std(ddof=1) / np.sqrt(len(t))) if len(t) > 1 else 0.0, len(t))
        if not row:
            continue
        acc[name] = row
        out[name] = {"infos": R[name]["infos"], "précision": {k: v[0] for k, v in row.items()},
                     "erreur type": {k: v[1] for k, v in row.items()}}
        print(f"  {name} ({R[name]['infos']['classes']} classes, {R[name]['infos']['entraînement']} exemples, "
              f"{R[name]['infos']['longueur']} pas × {R[name]['infos']['dimensions']})")
        for k, (m, se, s) in sorted(row.items(), key=lambda kv: -kv[1][0]):
            print(f"    {k:40s} {100 * m:6.2f} % ± {100 * se:.2f}  ({s} seeds)")
    for d in DATASETS:
        if d in R and "erreur" in R[d]:
            print(f"  {d} : non chargé ({R[d]['erreur'][:80]})")
    if not acc:
        return
    labels = [l for l, _, _ in MODELS if all(l in acc[d] for d in acc)]
    ranks = {l: [] for l in labels}
    for d in acc:
        order = sorted(labels, key=lambda l: -acc[d][l][0])
        for i, l in enumerate(order):
            ranks[l].append(i + 1)
    mean_rank = {l: float(np.mean(r)) for l, r in ranks.items()}
    mean_acc = {l: float(np.mean([acc[d][l][0] for d in acc])) for l in labels}
    print("  rang moyen (1 = meilleur) et précision moyenne sur les jeux :")
    for l in sorted(labels, key=lambda l: mean_rank[l]):
        print(f"    {l:40s} rang {mean_rank[l]:.2f}   précision moyenne {100 * mean_acc[l]:.2f} %")
    summary = {"jeux": out, "rang moyen": mean_rank, "précision moyenne": mean_acc}
    with open(core.out_path("data", f"uea_n{n}_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)

    ds = list(acc)
    M = np.array([[1 - acc[d][l][0] for l in labels] for d in ds]) * 100
    fig, ax = plt.subplots(figsize=(14, 0.5 * len(ds) + 3))
    rel = M / np.maximum(M.min(1, keepdims=True), 0.5)
    im = ax.imshow(np.log2(rel), cmap="RdYlGn_r", vmin=0, vmax=3, aspect="auto")
    for i in range(len(ds)):
        for j in range(len(labels)):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=8,
                    fontweight="bold" if M[i, j] == M[i].min() else "normal")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"{l}\nrang moyen {mean_rank[l]:.1f}" for l in labels], rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(len(ds))); ax.set_yticklabels(ds, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, label="log2(erreur / meilleure erreur de la ligne)")
    ax.set_title(f"Partie 8.7 — archive UEA, erreur de test en % ({n} neurones, learning rate choisi sur validation)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(core.out_path("figures", f"uea_n{n}.png"), dpi=110)
    print(f"\nFigure : results/figures/uea_n{n}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n, a.seeds, a.steps = 16, 2, 20
    tag = "_quick" if a.quick else ""
    path = core.out_path("data", f"uea_n{a.n}{tag}.json")
    res = {"args": {"n": a.n, "steps": a.steps, "synthetic": a.synthetic}, "results": {}}
    if path.exists():
        old = json.load(open(path))
        if old["args"] == res["args"]:
            res = old
    if not a.report:
        run(a, path, res)
    report(res, a.n)


if __name__ == "__main__":
    main()
