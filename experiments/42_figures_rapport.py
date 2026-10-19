"""Figures de synthèse du rapport, calculées à partir des résultats déjà enregistrés (aucun entraînement, quelques secondes).

    python experiments/42_figures_rapport.py

Sorties (results/figures/) :
    bilan_familles.png           facteur d'erreur de chaque famille contre le RNN, à 100 neurones, sur cinq tâches
    accorde_qui_apprend_quoi.png ce que valent les amplitudes g, les biais et les entrées W_in, seuls ou ensemble
    budget_curves.png            erreur contre paramètres appris : accordé de rang r, sans g, GRU et LRU de taille variable
    recherche_equitable.png      la recherche équitable (validation, 5 graines), tailles des modèles existants comprises
"""
import json

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from hfrnn import core

BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
FROZEN, INK, INK2, GRID, SURF = "#898781", "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f19a99", "#b42828"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
TASKS5 = ["Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)", "Lorenz (+10)", "Hénon (+3)"]
SHORT = {"Mackey-Glass (+15)": "Mackey-Glass", "NARMA10": "NARMA10", "Mémoire (délai 15)": "Mémoire",
         "Lorenz (+10)": "Lorenz", "Hénon (+3)": "Hénon", "Wiener-Hammerstein": "Wiener-Hammerstein"}


def load(name):
    return json.load(open(core.out_path("data", name)))


def save(fig, name):
    fig.savefig(core.out_path("figures", name), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("results/figures/" + name)


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def nmse(cell):
    return np.array([v["nmse"] for v in cell.values()])


def paired_gap(R, model, ref):
    """Écart apparié moyen log10(NMSE / NMSE_ref) sur les graines communes."""
    s = sorted(set(R[model]) & set(R[ref]), key=int)
    return float(np.mean([np.log10(R[model][k]["nmse"] / R[ref][k]["nmse"]) for k in s])) if s else np.nan


# ------------------------------------------------------------------ 1. bilan des familles
def fig_bilan_familles():
    """Écart apparié log10 au RNN (même réservoir, mêmes fenêtres), 100 neurones.
    Mackey-Glass, NARMA10, mémoire : expériences numpy à 8 graines (5 pour la mémoire) ;
    Lorenz et Hénon : expériences PyTorch à 3 graines (même protocole, RNN au learning rate standard)."""
    comp, sm, rot = load("comparison.json"), load("supermask.json"), load("rotation.json")["comparaison"]
    dec, cpl = load("decompositions_n100.json")["results"], load("coupled_forms_n100.json")["results"]

    def numpy_d(src, label):
        out = {}
        for t in TASKS5[:3]:
            rows = src[t]["table"] if isinstance(src[t], dict) else src[t]
            out[t] = next((r["d"] for r in rows if r["label"] == label), np.nan)
        return out

    def torch_d(R, label, ref="RNN", tasks=TASKS5):
        return {t: paired_gap(R[t], label, ref) if t in R and label in R[t] else np.nan for t in tasks}

    rows = [
        ("1 · RNN freiné", numpy_d(comp, "RNN freiné")),
        ("1 · mélange réglé (α appris)", numpy_d(comp, "mélange réglé (α appris)")),
        ("1 · réservoir corrigé de rang 4", numpy_d(comp, "réservoir corrigé de rang 4")),
        ("2 · réservoir sculpté intégral", numpy_d(sm, "réservoir sculpté intégral, dense")),
        ("3 · réservoir tourné", {**numpy_d(rot, "réservoir tourné (W_in tourné)"),
                                  **torch_d(dec, "réservoir tourné (rotations)", tasks=TASKS5[3:])}),
        ("4 · réservoir accordé", {**numpy_d(rot, "réservoir accordé"),
                                   **torch_d(dec, "réservoir accordé (valeurs singulières)", tasks=TASKS5[3:])}),
        ("5 · réservoir cisaillé", torch_d(dec, "Iwasawa : cisaillement (N)")),
        ("6 · réservoir nilpotent", torch_d(dec, "Jordan-Chevalley + partie nilpotente apprise")),
        ("7 · relais aléatoire", torch_d(cpl, "forme 3 : relais aléatoire")),
        ("7 · accordé + readout carré", torch_d(cpl, "contrôle : réservoir accordé, readout carré")),
        ("7 · duo croisé", torch_d(cpl, "forme 2 : duo croisé, lu côté réservoir")),
    ]
    M = np.array([[r[1].get(t, np.nan) for t in TASKS5] for r in rows])
    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    norm = TwoSlopeNorm(vmin=-1.8, vcenter=0, vmax=1.8)
    ax.imshow(np.ma.masked_invalid(M), cmap=DIV, norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fc="#f3f2ee", ec="none"))
                ax.text(j, i, "non mesuré", ha="center", va="center", fontsize=7.5, color=FROZEN)
            else:
                f = 10 ** (-v)
                lab = f"×{f:.1f}" if f >= 1 else f"÷{1 / f:.1f}"
                ax.text(j, i, lab, ha="center", va="center", fontsize=9,
                        color="white" if abs(v) > 1.0 else INK, weight="bold" if abs(v) > 0.3 else None)
    ax.set_xticks(range(5)); ax.set_xticklabels([SHORT[t] for t in TASKS5])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Chaque famille contre le RNN qui apprend tout (100 neurones)\n"
                 "×k : k fois moins d'erreur que le RNN (bleu) ; ÷k : k fois plus (rouge)", fontsize=11.5, weight="bold")
    fig.text(0.01, -0.03, "Mackey-Glass, NARMA10, mémoire : 8 graines (5 pour la mémoire), RNN au learning rate standard. "
             "Lorenz, Hénon : 3 graines, même protocole.\nLe numéro est celui de la partie du rapport.",
             fontsize=8, color=INK2)
    save(fig, "bilan_familles.png")


# ------------------------------------------------------------------ 2. qui apprend quoi dans l'accordé
def fig_qui_apprend_quoi():
    R = load("budget_inputs.json")["results"]
    models = [("réservoir classique", "rien (réservoir classique)", FROZEN),
              ("accordé : amplitudes g seules", "g seules", BLUE),
              ("contrôle : biais seul, sans g", "biais seuls", ORANGE),
              ("contrôle : entrées W_in seules, sans g", "W_in seules", ORANGE),
              ("contrôle : biais + entrées W_in, sans g", "biais + W_in (sans g)", ORANGE),
              ("accordé : g + biais", "g + biais", BLUE),
              ("accordé : g + entrées W_in", "g + W_in", BLUE),
              ("accordé : g + biais + entrées W_in", "g + biais + W_in (accordé)", BLUE)]
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.6), sharey=True)
    for ax, t in zip(axes, TASKS5):
        vals = [np.exp(np.mean(np.log(nmse(R[t][m])))) if m in R[t] else np.nan for m, _, _ in models]
        y = np.arange(len(models))[::-1]
        ax.axvline(vals[0], color=FROZEN, ls="--", lw=1.2)                     # niveau du réservoir classique
        for yi, v, (_, _, c) in zip(y, vals, models):
            ax.plot([vals[0], v], [yi, yi], color=c, lw=2.5, alpha=0.45)
            ax.plot(v, yi, "o", color=c, ms=9, mec=INK if yi in (y[4], y[7]) else c, mew=1.6)
        ax.set_xscale("log"); ax.set_title(SHORT[t], fontsize=11, weight="bold")
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks(y); ax.set_yticklabels([lab for _, lab, _ in models])
        ax.grid(axis="x", color=GRID, lw=0.8); clean(ax)
        ax.set_ylim(-0.6, len(models) - 0.4)
        ax.set_xlabel("NMSE (log, à gauche = mieux)")
    fig.suptitle("Dans le réservoir accordé, qui apprend quoi ? (200 neurones, 3 graines)", fontsize=13, weight="bold",
                 y=1.03)
    fig.text(0.5, -0.06, "■ bleu : les amplitudes singulières g apprennent      ■ orange : g figées à 1 (la récurrence reste "
             "exactement celle du réservoir)      ● gris et tirets : réservoir classique, seule la lecture apprend", ha="center", fontsize=9,
             color=INK2)
    save(fig, "accorde_qui_apprend_quoi.png")


# ------------------------------------------------------------------ 3. courbes à budget égal
def fig_budget_curves():
    R = load("budget_inputs.json")["results"]
    fams = [("accordé, N=500, rang ", "accordé + correction de rang r", BLUE, "o", "-"),
            ("sans g, N=500, rang ", "biais + W_in + rang r (sans g)", ORANGE, "s", "-"),
            ("GRU, ", "GRU, 20 à 500 unités", INK, "^", "-"),
            ("LRU, ", "LRU, 20 à 500 unités", FROZEN, "v", "--")]
    tasks = TASKS5 + ["Wiener-Hammerstein"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))
    for ax, t in zip(axes.flat, tasks):
        for pre, lab, c, mk, ls in fams:
            pts = []
            for m, cell in R.get(t, {}).items():
                if m.startswith(pre) and cell:
                    p = next(iter(cell.values()))["n_params"]
                    pts.append((p, np.exp(np.mean(np.log(nmse(cell))))))
            if pre == "accordé, N=500, rang " and t in ("Mackey-Glass (+15)", "Lorenz (+10)"):
                pts = [q for q in pts if q[0] < 200_000]     # le rang 256 diverge sur ces deux tâches : retiré du tracé
            if pre == "sans g, N=500, rang " and t == "Lorenz (+10)":
                pts = [q for q in pts if q[0] < 100_000]     # le rang 128 sans g diverge sur Lorenz : retiré du tracé
            if pts:
                pts.sort()
                ax.plot(*zip(*pts), marker=mk, color=c, ls=ls, lw=2, ms=6, label=lab)
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(SHORT[t], fontsize=11, weight="bold")
        ax.set_xlabel("paramètres appris (log)"); ax.set_ylabel("NMSE de test (log)")
        ax.grid(color=GRID, lw=0.8); clean(ax)
    axes.flat[0].legend(frameon=False, fontsize=8.5)
    fig.suptitle("À budget de paramètres égal : accordé de rang r, version sans g, GRU et LRU (3 graines)",
                 fontsize=13, weight="bold")
    fig.text(0.5, -0.01, "Le rang 256 (257 500 paramètres) diverge sur Mackey-Glass et Lorenz : il n'est pas tracé. "
             "Le rang 128 sans g diverge sur Lorenz : il n'est pas tracé.", ha="center", fontsize=8.5, color=INK2)
    fig.tight_layout()
    save(fig, "budget_curves.png")


# ------------------------------------------------------------------ 4. recherche équitable
def fig_recherche_equitable():
    F = load("fair_search_n500.json")["final"]
    order = ["GRU", "GRU, taille cherchée", "LRU", "LRU, taille cherchée", "réservoir accordé",
             "réservoir accordé à porte d'entrée", "réservoir à biais et entrées appris (sans g)"]
    colors = {"GRU": INK, "GRU, taille cherchée": "#3d3d3a", "LRU": FROZEN, "LRU, taille cherchée": "#b5b3ac",
              "réservoir accordé": BLUE, "réservoir accordé à porte d'entrée": AQUA,
              "réservoir à biais et entrées appris (sans g)": ORANGE}
    tasks = list(F)
    fig, axes = plt.subplots(1, len(tasks), figsize=(5.6 * len(tasks), 4.8), sharey=False)
    for ax, t in zip(np.atleast_1d(axes), tasks):
        ms = [m for m in order if m in F[t] and F[t][m]["seeds"]]
        L = [np.log10([v["nmse"] for v in F[t][m]["seeds"].values()]) for m in ms]
        y = np.arange(len(ms))[::-1]
        for yi, m, l in zip(y, ms, L):
            mu, se = l.mean(), l.std(ddof=1) / np.sqrt(len(l))
            ax.plot([10 ** (mu - se), 10 ** (mu + se)], [yi, yi], color=colors[m], lw=2)
            ax.plot(10 ** mu, yi, "o", color=colors[m], ms=10)
            ax.plot(10 ** l, [yi] * len(l), "|", color=colors[m], ms=8, alpha=0.6)
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_ylim(-0.6, len(ms) - 0.4)
        for yi, m, l in zip(y, ms, L):
            c = F[t][m]["cfg"]; p = next(iter(F[t][m]["seeds"].values()))["n_params"]
            extra = f", {c['n']} unités" if "n" in c else ""
            ax.text(10 ** l.max() * 1.6, yi, f"{p:,} par.{extra}".replace(",", " "), va="center", fontsize=7.5,
                    color=INK2)
        ax.set_yticks(y); ax.set_yticklabels(ms, fontsize=8.5); ax.set_xscale("log")
        ax.set_title(SHORT.get(t, t), fontsize=11, weight="bold"); ax.grid(axis="x", color=GRID, lw=0.8); clean(ax)
        ax.set_xlabel("NMSE de test (log) : point = moyenne, traits = les 5 graines")
        ax.set_xlim(right=ax.get_xlim()[1] * 30)
    fig.suptitle("Même budget de recherche pour tous : 12 essais choisis sur validation, test mesuré une fois "
                 "(500 neurones)", fontsize=12.5, weight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "recherche_equitable.png")


# ------------------------------------------------------------------ 5. cisaillé et nilpotent à 500 neurones
def fig_cisaille_500():
    R = load("decompositions_n500.json")["results"]
    ref = "réservoir accordé (valeurs singulières)"
    models = [("Iwasawa : cisaillement (N)", "cisaillé plein (125 750 par.)", "#9ec5f4"),
              ("réservoir cisaillé en bande, k = 1", "cisaillé en bande, k = 1 (1 499)", "#86b6ef"),
              ("réservoir cisaillé en bande, k = 3", "cisaillé en bande, k = 3 (2 494)", BLUE),
              ("réservoir cisaillé en bande, k = 10", "cisaillé en bande, k = 10 (5 945)", "#104281"),
              ("Jordan-Chevalley + partie nilpotente apprise", "nilpotent plein (126 750)", "#f5b08f"),
              ("réservoir nilpotent en bande, k = 3", "nilpotent en bande, k = 3 (3 494)", ORANGE)]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    w = 0.13
    for i, (m, lab, c) in enumerate(models):
        g = [-paired_gap(R[t], m, ref) for t in TASKS5]
        ax.bar(np.arange(5) + (i - 2.5) * w, g, width=w, color=c, label=lab)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(range(5)); ax.set_xticklabels([SHORT[t] for t in TASKS5])
    ax.set_ylim(-2.4, 1.4)
    ax.set_yticks([-2, -1, -np.log10(3), 0, np.log10(3), 1])
    ax.set_yticklabels(["÷100", "÷10", "÷3", "égal", "×3", "×10"])
    ax.set_ylabel("facteur d'erreur contre l'accordé\n(au-dessus de 0 = mieux que lui)")
    ax.grid(axis="y", color=GRID, lw=0.8); clean(ax)
    ax.legend(frameon=False, fontsize=8.5, ncol=2, loc="lower left")
    ax.set_title("Cisaillé et nilpotent à 500 neurones, pleins ou en bande, contre le réservoir accordé (1 500 par., 3 graines)",
                 fontsize=11.5, weight="bold")
    save(fig, "cisaille_nilpotent_500.png")


if __name__ == "__main__":
    fig_bilan_familles(); fig_qui_apprend_quoi(); fig_budget_curves(); fig_recherche_equitable(); fig_cisaille_500()
