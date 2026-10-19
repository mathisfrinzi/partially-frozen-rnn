"""
Partie 8.3 — MNIST pixel par pixel : 784 pas par image, un pixel à la fois.

Lire l'image ligne par ligne (28 pas, partie 8.1) demande peu de mémoire. Pixel par pixel, l'information utile est
étalée sur des centaines de pas : c'est là qu'un réservoir, dont la mémoire s'efface en quelques dizaines de pas
(partie 3.4), risque d'échouer, et là que les récurrences linéaires de type LRU sont réputées fortes. Un test honnête
de la limite de l'approche.

Modèles (128 neurones) : RNN, GRU, LRU, réservoir accordé, réservoir accordé linéaire (type LRU). Chaque modèle
choisit son learning rate (1e-3 ou 3e-3) sur 10 000 images de validation avec la seed 0, puis le learning rate
retenu est ré-entraîné sur deux autres seeds ; la précision de test n'est mesurée qu'à la fin de chaque époque.

    python experiments/27_pixel_mnist.py                     # GPU (~1 h 15 sur un T4)
    python experiments/27_pixel_mnist.py --quick --synthetic  # vérification sur CPU
    python experiments/27_pixel_mnist.py --report
Sorties : results/data/pixel_mnist.json, results/figures/pixel_mnist.png
"""
import argparse
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn import core
from hfrnn import torch_models as TM

_spec = importlib.util.spec_from_file_location("mnist_search", Path(__file__).with_name("23_mnist_search.py"))
M = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(M)

MODELS = {
    "RNN": ({"kind": "convex", "alpha": 1.0}, "tab:red"),
    "GRU": ({"kind": "gru", "rec_init": "ortho"}, "tab:orange"),
    "LRU": ({"kind": "lru"}, "tab:pink"),
    "réservoir accordé": ({"kind": "spectrum"}, "tab:blue"),
    "réservoir accordé linéaire (type LRU)": ({"kind": "linear_tuned"}, "tab:green"),
}
LRS = [1e-3, 3e-3]


def to_pixels(data):
    """(N, 28, 28) → (N, 784, 1) : un pixel par pas."""
    return {k: (v.reshape(len(v), -1, 1) if k.startswith("X") else v) for k, v in data.items()}


def fit(data, m, lr, seed, a, dev):
    kw = dict(MODELS[m][0])
    kind = kw.pop("kind")
    return TM.fit_classifier(data, kind, seed, n=a.n, epochs=a.epochs, lr=lr, batch=a.batch, device=dev, test=True,
                             rho=0.9, in_scale=0.5, density=0.1, **kw)


def run(a, path, res):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.synthetic:
        data = M.synthetic_data(a.train, a.val, a.test, device=dev)
    else:
        data = M.mnist_data(a.val, dev, limit=a.train if a.quick else None)
    data = to_pixels(data)
    print(f"appareil : {dev} | N = {a.n} | {a.epochs} époques | séquences de {data['Xtr'].shape[1]} pas", flush=True)
    t0 = time.time()

    def go(m, lr, seed):
        cell = res["runs"].setdefault(m, {}).setdefault(f"{lr:g}", {})
        if str(seed) in cell:
            return cell[str(seed)]
        r = fit(data, m, lr, seed, a, dev)
        cell[str(seed)] = r
        print(f"[{(time.time() - t0) / 60:6.1f} min] {m:38s} lr {lr:g} seed {seed}  validation {r['val'][-1]:.4f}  "
              f"test {r['test'][-1]:.4f}{' (divergé)' if r['diverged'] else ''}  ({r['seconds'] / 60:.1f} min)", flush=True)
        with open(path, "w") as f:
            json.dump(res, f, indent=1, ensure_ascii=False)
        return r
    for m in MODELS:                                           # 1. learning rate choisi sur la validation (seed 0)
        for lr in LRS:
            go(m, lr, 0)
    for m in MODELS:                                           # 2. le learning rate retenu, sur d'autres seeds
        lr = best_lr(res, m)
        for seed in range(1, a.seeds):
            go(m, lr, seed)


def best_lr(res, m):
    return max(LRS, key=lambda lr: res["runs"][m][f"{lr:g}"]["0"]["val"][-1])


def report(res):
    a = res["args"]
    out = {}
    print(f"\nMNIST pixel par pixel, N = {a['n']}, {a['epochs']} époques ; learning rate choisi sur la validation")
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    for i, (m, (_, c)) in enumerate(MODELS.items()):
        if m not in res["runs"]:
            continue
        lr = best_lr(res, m)
        runs = res["runs"][m][f"{lr:g}"]
        err = np.array([1 - r["test"][-1] for r in runs.values()]) * 100
        se = err.std(ddof=1) / np.sqrt(len(err)) if len(err) > 1 else 0.0
        npar = next(iter(runs.values()))["n_params"]
        out[m] = {"lr": lr, "erreur test (%)": float(err.mean()), "se": float(se), "seeds": len(err), "n_params": npar}
        print(f"  {m:38s} lr {lr:g}  erreur test {err.mean():6.2f} % ± {se:.2f}  ({len(err)} seeds, {npar} paramètres "
              f"appris hors lecture)")
        axes[0].bar(i, err.mean(), yerr=se, color=c, capsize=3)
        curve = np.mean([1 - np.array(r["test"]) for r in runs.values()], 0) * 100
        axes[1].plot(range(1, len(curve) + 1), curve, marker="o", color=c, label=m)
    axes[0].set_xticks(range(len(MODELS))); axes[0].set_xticklabels(list(MODELS), rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("erreur de test (%)"); axes[0].set_yscale("log")
    axes[0].set_title(f"après {a['epochs']} époques")
    axes[1].set_xlabel("époque"); axes[1].set_ylabel("erreur de test (%)"); axes[1].set_yscale("log")
    axes[1].legend(fontsize=8); axes[1].set_title("apprentissage")
    fig.suptitle(f"Partie 8.3 — MNIST pixel par pixel (784 pas), {a['n']} neurones", fontsize=12)
    fig.tight_layout()
    tag = "_quick" if a.get("quick") else ""
    fig.savefig(core.out_path("figures", f"pixel_mnist{tag}.png"), dpi=110)
    with open(core.out_path("data", f"pixel_mnist{tag}_summary.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    print(f"\nFigure : results/figures/pixel_mnist{tag}.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--val", type=int, default=10000)
    ap.add_argument("--train", type=int, default=50000)
    ap.add_argument("--test", type=int, default=10000)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.n, a.epochs, a.seeds, a.batch, a.val, a.train, a.test = 16, 1, 2, 64, 200, 400, 200
    tag = "_quick" if a.quick else ""
    path = core.out_path("data", f"pixel_mnist{tag}.json")
    args = {k: getattr(a, k) for k in ("n", "epochs", "batch", "val", "quick", "synthetic")}
    res = {"args": args, "runs": {}}
    if path.exists():
        old = json.load(open(path))
        if old["args"] == args:
            res = old
    if not a.report:
        run(a, path, res)
    report(res)


if __name__ == "__main__":
    main()
