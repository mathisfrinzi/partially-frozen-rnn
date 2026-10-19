"""
GPU : une tâche plus réaliste, MNIST séquentiel (une image = 28 pas de 28 pixels).

Classification à 10 classes ; le readout lit le dernier état et l'état moyen.
Modèles : réservoir classique (readout seul), RNN, mélange fixe α ∈ {0.1, 0.3}, RNN freiné,
réservoir corrigé de rang 4 et 16, réservoir retouché, mélange réglé (α appris).

    python experiments/06_gpu_smnist.py                    # N = 256, 3 époques, 3 seeds
    python experiments/06_gpu_smnist.py --quick            # vérification rapide
"""
import argparse
import json
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from hfrnn import core
from hfrnn import torch_models as TM

MODELS = [
    ("réservoir classique", {"kind": "reservoir"}),
    ("RNN", {"kind": "convex", "alpha": 1.0}),
    ("mélange fixe α = 0.1", {"kind": "convex", "alpha": 0.1}),
    ("mélange fixe α = 0.3", {"kind": "convex", "alpha": 0.3}),
    ("RNN ralenti × 0.1", {"kind": "convex", "alpha": 1.0, "lr_mult": 0.1}),
    ("RNN freiné", {"kind": "convex", "alpha": 1.0, "lr_mult": 0.1, "lr_mult_in": 1.0}),
    ("réservoir corrigé de rang 4", {"kind": "lowrank", "rank": 4}),
    ("réservoir corrigé de rang 16", {"kind": "lowrank", "rank": 16}),
    ("réservoir retouché", {"kind": "sparse"}),
    ("mélange réglé (α appris)", {"kind": "lalpha", "alpha0": 0.1}),
    # partie 2 : supermasque (les poids aléatoires ne changent jamais)
    ("réservoir classique, dense", {"kind": "reservoir", "sparsity": 1.0}),
    ("RNN, départ dense", {"kind": "convex", "alpha": 1.0, "sparsity": 1.0}),
    ("réservoir sculpté, dense", {"kind": "supermask", "mode": "sigmoid", "win": "free", "mult": 50.0, "sparsity": 1.0}),
    ("réservoir sculpté intégral, dense",
     {"kind": "supermask", "mode": "sigmoid", "win": "mask", "mult": 50.0, "sparsity": 1.0}),
    # partie 3 : spectre contre directions
    ("réservoir tourné", {"kind": "rotation", "win": "rotate"}),
    ("réservoir accordé", {"kind": "spectrum"}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--in-scale", type=float, default=0.1)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.seeds = 1, 1
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tf = transforms.ToTensor()
    train = datasets.MNIST("data_mnist", train=True, download=True, transform=tf)
    test = datasets.MNIST("data_mnist", train=False, download=True, transform=tf)
    if a.quick:
        train = torch.utils.data.Subset(train, range(5000))
        test = torch.utils.data.Subset(test, range(2000))
    tl = torch.utils.data.DataLoader(train, batch_size=a.batch, shuffle=True, num_workers=2)
    vl = torch.utils.data.DataLoader(test, batch_size=1000, num_workers=2)
    print(f"appareil : {dev} | N = {a.n} | {a.epochs} époques | {a.seeds} seeds", flush=True)

    tag = f"n{a.n}" + ("_quick" if a.quick else "")
    path = core.out_path("data", f"gpu_smnist_{tag}.json")
    res, curves = {}, {}
    if path.exists():                       # on complète les résultats déjà calculés avec les mêmes réglages
        old = json.load(open(path))
        if all(old["args"].get(k) == getattr(a, k) for k in ("n", "epochs", "seeds", "in_scale", "rho", "batch")):
            res, curves = old["accuracy"], old["curves"]
            print("résultats existants repris :", list(res), flush=True)
    t0 = time.time()
    for label, kw in MODELS:
        if label in res:
            continue
        accs, hs = [], []
        kw2 = dict(kw)
        sparsity = kw2.pop("sparsity", core.SPARSITY)
        for seed in range(a.seeds):
            Win, W = TM.make_reservoir(a.n, n_in=28, rho=a.rho, in_scale=a.in_scale, sparsity=sparsity, seed=seed)
            acc, hist = TM.train_classifier(tl, vl, Win, W, seed, epochs=a.epochs, device=dev,
                                            log=lambda m: print(m, flush=True), **kw2)
            accs.append(acc); hs.append(hist)
        res[label], curves[label] = accs, hs
        print(f"[{time.time() - t0:6.0f}s] {label:40s} précision = {np.mean(accs):.4f} "
              f"(min {np.min(accs):.4f}, max {np.max(accs):.4f})", flush=True)
        with open(path, "w") as f:
            json.dump({"args": vars(a), "accuracy": res, "curves": curves}, f, indent=1)

    plot(res, curves, a.n, a.epochs, tag)


def plot(res, curves, n, epochs, tag):
    labels = [l for l, _ in MODELS if l in res]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
    err = [1 - np.mean(res[l]) for l in labels]
    axes[0].bar(range(len(labels)), err, color="tab:purple")
    axes[0].set_xticks(range(len(labels))); axes[0].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    axes[0].set_yscale("log"); axes[0].set_ylabel("taux d'erreur test (log)")
    axes[0].set_title(f"MNIST séquentiel (28 × 28), N = {n}")
    for l in labels:
        axes[1].plot(range(1, epochs + 1), 1 - np.mean(curves[l], axis=0), marker="o", label=l)
    axes[1].set_yscale("log"); axes[1].set_xlabel("époque"); axes[1].set_ylabel("taux d'erreur test")
    axes[1].legend(fontsize=8); axes[1].set_title("Apprentissage")
    fig.tight_layout()
    fig.savefig(core.out_path("figures", f"gpu_smnist_{tag}.png"), dpi=110)
    print(f"\nFigure : results/figures/gpu_smnist_{tag}.png")


if __name__ == "__main__":
    main()
