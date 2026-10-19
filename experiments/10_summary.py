"""
Figure de synthèse des trois parties (à partir des résultats de 02, 08 et 09).

    python experiments/10_summary.py      -> results/figures/summary.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn.core import out_path

TASKS = ["Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)"]


def main():
    comp = json.load(open(out_path("data", "comparison.json")))
    sm = json.load(open(out_path("data", "supermask.json")))
    rot = json.load(open(out_path("data", "rotation.json")))["comparaison"]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    for ax, t in zip(axes, TASKS):
        c = {r["label"]: r for r in comp[t]}
        s = {r["label"]: r for r in sm[t]["table"]}
        r = {x["label"]: x for x in rot[t]}
        best = min(("RNN", "RNN, lr global × 0.3", "RNN, lr global × 0.1"), key=lambda k: c[k]["mean"])
        best_rot = min(("réservoir tourné (W_in tourné)", "réservoir accordé"),
                       key=lambda k: r[k]["mean"])
        bars = [("réservoir\nclassique", c["réservoir classique"], "gray"),
                ("meilleur\nRNN", c[best], "tab:red"),
                ("P1\nRNN freiné", c["RNN freiné"], "tab:brown"),
                ("P2\nréservoir sculpté\nintégral", s["réservoir sculpté intégral, dense"],
                 "tab:purple"),
                ("P3\n" + ("réservoir\ntourné" if "tourné" in best_rot else "réservoir\naccordé") + "\n(meilleur des 2)",
                 r[best_rot], "tab:green")]
        ax.bar(range(len(bars)), [b[1]["mean"] for b in bars], yerr=[b[1]["se"] for b in bars], capsize=4,
               color=[b[2] for b in bars])
        for i, (_, b, _) in enumerate(bars[2:], start=2):
            ax.text(i, (b["mean"] + b["se"]) * 1.25, f"×{c[best]['mean'] / b['mean']:.1f}", ha="center",
                    va="bottom", fontsize=11, fontweight="bold")
        means = [b[1]["mean"] for b in bars]
        ax.set_ylim(min(means) / 3, max(means) * 3)
        ax.set_xticks(range(len(bars))); ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
        ax.set_yscale("log"); ax.set_ylabel("erreur de test (NMSE, log)"); ax.set_title(t, fontsize=12)
    fig.suptitle("Plus bas = mieux. « ×k » : k fois moins d'erreur que le meilleur RNN. "
                 "P1, P2, P3 : parties 1, 2 et 3.", fontsize=11)
    fig.tight_layout(); fig.savefig(out_path("figures", "summary.png"), dpi=120)
    print("Figure : results/figures/summary.png")


if __name__ == "__main__":
    main()
