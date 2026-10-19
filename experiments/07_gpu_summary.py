"""
Figure de synthèse des expériences GPU (05 et 06), à partir de leurs fichiers de résultats.

    python experiments/07_gpu_summary.py

Lit results/data/gpu_regression_n500.json et results/data/gpu_smnist_n256.json,
écrit results/figures/gpu_summary.png et affiche les écarts appariés.
"""
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn.core import out_path

COLORS = {"réservoir classique": "gray", "RNN": "tab:red", "réservoir retouché": "tab:purple", "mélange réglé (α appris)": "tab:cyan",
          "RNN freiné": "tab:brown"}


def color(label):
    if label in COLORS:
        return COLORS[label]
    if label.startswith(("réservoir sculpté", "réservoir élagué")):
        return "tab:purple"
    if label.startswith("réservoir tourné"):
        return "tab:green"
    if label.startswith("réservoir accordé"):
        return "tab:blue"
    if "+" in label or label.startswith("accordé"):                 # partie 4 : combinaisons
        return "black"
    if label == "réservoir classique, dense":
        return "silver"
    if label == "RNN, départ dense":
        return "salmon"
    if label.startswith("RNN, lr global"):
        return "salmon"
    if label.startswith("mélange"):
        return "tab:pink"
    if label.startswith("RNN ralenti"):
        return "tab:orange"
    if label.startswith("réservoir corrigé"):
        return "tab:olive"
    return "black"


def paired(a, ref):
    d = np.log10(np.asarray(a) / np.asarray(ref))
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


# ordre d'affichage : références, partie 1, partie 2, partie 3
ORDER = ["réservoir classique", "réservoir classique, dense", "RNN", "RNN, départ dense", "RNN, lr global × 0.1",
         "RNN, lr global × 0.03", "RNN, lr global × 0.01",
         "RNN ralenti × 0.1", "RNN freiné", "mélange fixe α = 0.1", "mélange fixe α = 0.3", "mélange réglé (α appris)",
         "réservoir corrigé de rang 4", "réservoir corrigé de rang 16", "réservoir retouché",
         "réservoir sculpté, dense", "réservoir sculpté intégral, dense", "réservoir tourné", "réservoir accordé",
         "réservoir accordé, dense", "accordé intégral", "accordé + sculpté, dense", "accordé + sculpté intégral, dense",
         "accordé + freiné, dense", "sculpté + freiné, dense", "accordé + sculpté + freiné, dense"]


def ordered(labels):
    return [l for l in ORDER if l in labels] + [l for l in labels if l not in ORDER]


def headline(reg, sm):
    """Figure courte : les meilleurs représentants de chaque partie, comme summary.png mais à grande échelle."""
    bars = [("réservoir\nclassique", "réservoir classique", "gray"), ("meilleur\nRNN", None, "tab:red"),
            ("P1\nRNN freiné", "RNN freiné", "tab:brown"),
            ("P2\nréservoir sculpté\nintégral", "réservoir sculpté intégral, dense", "tab:purple"),
            ("P3\nréservoir\naccordé", "réservoir accordé", "tab:blue")]
    nt = len(reg["results"])
    fig, axes = plt.subplots(1, nt + 1, figsize=(6.4 * (nt + 1), 5.8))
    for ax, (task, res) in zip(axes, reg["results"].items()):
        if "réservoir accordé" not in res:
            continue
        best = min((l for l in res if l == "RNN" or l.startswith("RNN, lr global")), key=lambda l: np.mean(res[l]))
        labs = [best if l is None else l for _, l, _ in bars]
        m = [np.mean(res[l]) for l in labs]
        se = [np.std(res[l], ddof=1) / np.sqrt(len(res[l])) for l in labs]
        ax.bar(range(len(labs)), m, yerr=se, capsize=4, color=[c for _, _, c in bars])
        for i, l in enumerate(labs[2:], start=2):
            ax.text(i, (m[i] + se[i]) * 1.3, f"×{np.mean(res[best]) / m[i]:.0f}", ha="center", va="bottom",
                    fontsize=11, fontweight="bold")
        ax.set_ylim(min(m) / 3, max(m) * 3)
        ax.set_xticks(range(len(labs))); ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
        ax.set_yscale("log"); ax.set_ylabel("erreur de test (NMSE, log)")
        ax.set_title(f"{task}, N = {reg['args']['n']}", fontsize=12)
    ax = axes[nt]
    acc = sm["accuracy"]
    labs = ["réservoir classique", "RNN", "RNN freiné", "réservoir sculpté intégral, dense", "réservoir accordé"]
    err = [100 * (1 - np.mean(acc[l])) for l in labs]
    se = [100 * np.std(acc[l], ddof=1) / np.sqrt(len(acc[l])) for l in labs]
    ax.bar(range(len(labs)), err, yerr=se, capsize=4, color=[c for _, _, c in bars])
    for i in range(2, len(labs)):
        ax.text(i, err[i] + se[i] + 0.4, f"{err[i]:.1f} %", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(1, err[1] + se[1] + 0.4, f"{err[1]:.1f} %", ha="center", va="bottom", fontsize=11)
    ax.set_ylim(0, 12)
    ax.text(0, 11.5, f"↑ {err[0]:.0f} %", ha="center", va="top", fontsize=10, color="white",
            fontweight="bold")
    ax.set_xticks(range(len(labs))); ax.set_xticklabels([b[0].replace("meilleur\n", "") for b in bars], fontsize=8.5)
    ax.set_ylabel("taux d'erreur de test (%)")
    ax.set_title(f"MNIST séquentiel, N = {sm['args']['n']}, {sm['args']['epochs']} époques", fontsize=12)
    fig.suptitle("Grands réseaux (GPU, 3 seeds). Plus bas = mieux. « ×k » : k fois moins d'erreur que le meilleur RNN.",
                 fontsize=11)
    fig.tight_layout(); fig.savefig(out_path("figures", "gpu_headline.png"), dpi=120)
    print("Figure : results/figures/gpu_headline.png")


def main():
    reg = json.load(open(out_path("data", "gpu_regression_n500.json")))
    sm = json.load(open(out_path("data", "gpu_smnist_n256.json")))
    n, seeds = reg["args"]["n"], reg["args"]["seeds"]

    nt = len(reg["results"])
    fig, axes = plt.subplots(1, nt + 1, figsize=(6.6 * (nt + 1), 5.6))
    for ax, (task, res) in zip(axes, reg["results"].items()):
        labels = ordered(list(res))
        m = [np.mean(res[l]) for l in labels]
        se = [np.std(res[l], ddof=1) / np.sqrt(len(res[l])) for l in labels]
        ax.bar(range(len(labels)), m, yerr=se, capsize=3, color=[color(l) for l in labels])
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
        ax.set_yscale("log"); ax.set_ylabel("erreur de test (NMSE, log)")
        ax.set_title(f"{task}, N = {n} ({seeds} seeds)")
        best_rnn = min((l for l in labels if l == "RNN" or l.startswith("RNN, lr global")), key=lambda l: np.mean(res[l]))
        print(f"\n{task}, N = {n} : écart apparié log10(NMSE / {best_rnn})")
        for l in labels:
            d, s = paired(res[l], res[best_rnn])
            print(f"  {l:34s} {np.mean(res[l]):10.3g}   {d:+.2f} ± {s:.2f}   (×{10 ** -d:.1f})")
        ax.axhline(np.mean(res[best_rnn]), color="tab:red", ls="--", lw=1)

    ax = axes[nt]
    acc = sm["accuracy"]
    labels = ordered(list(acc))
    err = [100 * (1 - np.mean(acc[l])) for l in labels]
    e_se = [100 * np.std(acc[l], ddof=1) / np.sqrt(len(acc[l])) for l in labels]
    ax.bar(range(len(labels)), err, yerr=e_se, capsize=3, color=[color(l) for l in labels])
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("taux d'erreur de test (%, log)")
    ax.axhline(100 * (1 - np.mean(acc["RNN"])), color="tab:red", ls="--", lw=1)
    ax.set_title(f"MNIST séquentiel, N = {sm['args']['n']}, {sm['args']['epochs']} époques "
                 f"({sm['args']['seeds']} seeds)")
    print("\nMNIST séquentiel : taux d'erreur (%) et écart au RNN en points (seeds appariées)")
    ref = np.array(acc["RNN"])
    for l in labels:
        d = 100 * (np.array(acc[l]) - ref)
        print(f"  {l:34s} {100 * (1 - np.mean(acc[l])):6.2f} %   précision {d.mean():+.2f} ± "
              f"{d.std(ddof=1) / np.sqrt(len(d)):.2f} pts")
    fig.suptitle("Expériences GPU — plus bas = mieux. Tirets rouges : meilleur RNN.", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path("figures", "gpu_summary.png"), dpi=120)
    print("\nFigure : results/figures/gpu_summary.png")
    headline(reg, sm)


if __name__ == "__main__":
    main()
