"""Schémas pédagogiques : un dessin par famille de modèles (ce qui est figé, ce qui est appris, ce que ça fait).

Aucun entraînement : les figures montrent le principe de chaque modèle sur de petits exemples calculés (matrices de
12 neurones, transformations du plan, normes d'état). Quelques secondes sur CPU.

    python experiments/38_schemas.py

Code couleur commun à toutes les figures :
    gris   = figé (tiré au hasard, jamais appris)
    bleu   = appris
    orange = appris aussi, mais d'une autre nature (entrées, seconde pièce d'une décomposition)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle

from hfrnn import core

# palette (validée : bleu / orange / aqua, gris neutre, encres)
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
FROZEN, FROZEN_FILL = "#898781", "#ecebe6"
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
DIV = LinearSegmentedColormap.from_list("div", ["#1c5cab", "#86b6ef", "#f0efec", "#f19a99", "#b42828"])
SEQ = LinearSegmentedColormap.from_list("seq", ["#f0efec", "#9ec5f4", "#2a78d6", "#104281"])

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "text.color": INK, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def save(fig, name):
    fig.savefig(core.out_path("figures", name), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("results/figures/" + name)


# ---------------------------------------------------------------- briques de dessin
def box(ax, xy, w, h, text, learned, sub=None, color=None):
    c = color or (BLUE if learned else FROZEN)
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.08", lw=2,
                                ec=c, fc=(c + "22") if learned else FROZEN_FILL))
    ax.text(xy[0] + w / 2, xy[1] + h / 2 + (0.07 if sub else 0), text, ha="center", va="center", fontsize=11,
            color=INK, weight="bold")
    if sub:
        ax.text(xy[0] + w / 2, xy[1] + h / 2 - 0.13, sub, ha="center", va="center", fontsize=8, color=INK2)


def arrow(ax, a, b, learned=None, label=None, rad=0.0, lw=2, style="-|>", ls="-", lab_off=(0, 0.08)):
    c = INK2 if learned is None else (BLUE if learned else FROZEN)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14, lw=lw, color=c,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, shrinkA=2, shrinkB=2))
    if label:
        m = ((a[0] + b[0]) / 2 + lab_off[0], (a[1] + b[1]) / 2 + lab_off[1])
        ax.text(*m, label, ha="center", va="center", fontsize=9, color=c if learned is not None else INK2,
                bbox=dict(fc=SURF, ec="none", pad=0.5))


def blank(ax, xlim=(0, 1), ylim=(0, 1)):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal"); ax.axis("off")


def legend_frozen_learned(fig, y=0.0, extra=None):
    items = [(FROZEN, "figé (aléatoire, jamais appris)"), (BLUE, "appris")] + (extra or [])
    x = 0.5 - 0.17 * len(items) / 2
    for i, (c, t) in enumerate(items):
        fig.text(x + 0.17 * i, y, "■ ", color=c, fontsize=13, ha="left", va="center")
        fig.text(x + 0.17 * i + 0.017, y, t, color=INK2, fontsize=9, ha="left", va="center")


def mini_rnn(ax, title, w_in, w_rec, w_out, eq, rec_label="W", in_label="W_in"):
    """Réseau minimal u -> x (boucle) -> ŷ, chaque flèche colorée selon qu'elle est apprise."""
    blank(ax, (0, 3.2), (0, 2.5))
    ax.text(1.6, 2.45, title, ha="center", va="top", fontsize=12, weight="bold")
    ax.text(0.25, 1.0, "u(t)", ha="center", va="center", fontsize=11)
    ax.add_patch(Circle((1.6, 1.0), 0.38, fc=SURF, ec=INK2, lw=1.5))
    ax.text(1.6, 1.0, "x(t)", ha="center", va="center", fontsize=11)
    ax.text(2.95, 1.0, "ŷ(t)", ha="center", va="center", fontsize=11)
    arrow(ax, (0.45, 1.0), (1.2, 1.0), w_in, in_label)
    arrow(ax, (2.0, 1.0), (2.75, 1.0), w_out, "W_out")
    c = BLUE if w_rec is True else FROZEN if w_rec is False else ORANGE
    ax.add_patch(FancyArrowPatch((1.38, 1.36), (1.82, 1.36), arrowstyle="-|>", mutation_scale=14, lw=2, color=c,
                                 connectionstyle="arc3,rad=-2.2", shrinkA=2, shrinkB=2))
    ax.text(1.6, 2.0, rec_label, ha="center", va="center", fontsize=10, color=c)
    ax.text(1.6, 0.28, eq, ha="center", va="center", fontsize=9, color=INK2)


# ---------------------------------------------------------------- 1. le principe
def fig_principe():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    mini_rnn(axes[0], "Réservoir (ESN)", False, False, True, "seul le readout W_out apprend")
    mini_rnn(axes[1], "RNN classique", True, True, True, "tout apprend par rétropropagation")
    mini_rnn(axes[2], "Hybride (cette étude)", True, "mix", True, "W = W₀ (figé) + un peu d'apprentissage",
             rec_label="W₀ + ΔW")
    fig.suptitle("Deux extrêmes, et ce qu'on étudie entre les deux", fontsize=13, weight="bold", y=1.02)
    legend_frozen_learned(fig, y=-0.02, extra=[(ORANGE, "mélange : figé + appris")])
    save(fig, "schema_principe.png")


# ---------------------------------------------------------------- 2. partie 1 : mélange et freinage
def fig_partie1():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    names = ["RNN classique", "RNN ralenti × 0.1", "Mélange fixe α = 0.1", "RNN freiné"]
    lr_in = [1, 0.1, 0.1, 1]
    lr_rec = [1, 0.1, 0.1, 0.1]
    y = np.arange(len(names))[::-1]
    a1.barh(y + 0.18, lr_in, height=0.34, color=ORANGE, label="vitesse d'apprentissage des entrées W_in")
    a1.barh(y - 0.18, lr_rec, height=0.34, color=BLUE, label="vitesse d'apprentissage de la récurrence W")
    a1.set_yticks(y); a1.set_yticklabels(names); a1.set_xscale("log"); a1.set_xlim(0.05, 1.6)
    a1.set_xticks([0.1, 1]); a1.set_xticklabels(["× 0.1", "× 1"])
    a1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.38), frameon=False, fontsize=9, ncol=1)
    a1.set_title("Ce que fait vraiment chaque modèle (avec Adam)", fontsize=11, weight="bold")
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)
    a1.text(0.13, y[2], "← le mélange fixe (1−α)·W₀ + α·ΔW\n    revient à un RNN ralenti",
            fontsize=8.5, color=INK2, va="center")
    # droite : trajectoire schématique dans l'espace des poids
    rng = np.random.default_rng(3)
    a2.set_title("Rester près du réservoir", fontsize=11, weight="bold")
    W0 = np.array([0.0, 0.0])
    a2.add_patch(Circle(W0, 0.9, fc=FROZEN_FILL, ec=FROZEN, lw=1, ls="--"))
    a2.plot(*W0, "o", color=FROZEN, ms=9); a2.text(0.07, -0.18, "W₀ (réservoir)", color=INK2, fontsize=9)
    t = np.linspace(0, 1, 40)
    rnn = np.c_[2.6 * t + 0.25 * np.sin(9 * t), 1.9 * t ** 0.8 + 0.2 * np.sin(7 * t)]
    fre = np.c_[0.7 * t + 0.05 * np.sin(9 * t), 0.55 * t ** 0.8]
    a2.plot(rnn[:, 0], rnn[:, 1], color=INK2, lw=2); a2.plot(*rnn[-1], "o", color=INK2, ms=7)
    a2.text(*(rnn[-1] + [0.05, 0.08]), "RNN : s'éloigne,\nsur-apprend", fontsize=9, color=INK2)
    a2.plot(fre[:, 0], fre[:, 1], color=BLUE, lw=2.5); a2.plot(*fre[-1], "o", color=BLUE, ms=7)
    a2.text(*(fre[-1] + [0.08, -0.2]), "RNN freiné :\nreste dans la zone", fontsize=9, color=BLUE)
    a2.set_xlim(-1.1, 3.4); a2.set_ylim(-1.1, 2.6); a2.set_aspect("equal")
    a2.set_xticks([]); a2.set_yticks([]); a2.set_xlabel("espace des poids récurrents (schéma)")
    for s in a2.spines.values():
        s.set_color(GRID)
    fig.suptitle("Partie 1 — mélanger un réservoir et un RNN revient à freiner la récurrence", fontsize=13,
                 weight="bold", y=1.02)
    save(fig, "schema_partie1_freine.png")


# ---------------------------------------------------------------- 3. réservoir sculpté
def small_reservoir(n=12, seed=4, density=0.35):
    rng = np.random.default_rng(seed)
    W = rng.uniform(-1, 1, (n, n)) * (rng.random((n, n)) < density)
    return W / max(abs(np.linalg.eigvals(W))) * 0.9


def heat(ax, M, title, cmap=DIV, vmax=None, sub=None):
    v = vmax or np.abs(M).max()
    ax.imshow(M, cmap=cmap, vmin=-v if cmap is DIV else 0, vmax=v)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_title(title, fontsize=10.5, weight="bold")
    if sub:
        ax.set_xlabel(sub, fontsize=8.5, color=INK2)


def fig_sculpte():
    W0 = small_reservoir()
    rng = np.random.default_rng(1)
    m = np.clip(rng.beta(2.2, 1.0, W0.shape), 0, 1)               # masque d'atténuation appris (illustration)
    binary = (m > 0.55).astype(float)
    g = rng.normal(1, 0.9, W0.shape)
    v = np.abs(W0).max()
    fig, ax = plt.subplots(1, 5, figsize=(15, 3.6), gridspec_kw={"width_ratios": [1, 0.18, 1, 0.18, 1]})
    heat(ax[0], W0, "W₀ : réservoir aléatoire", vmax=v, sub="figé")
    ax[1].axis("off"); ax[1].text(0.5, 0.5, "⊙", fontsize=28, ha="center", va="center", color=INK2)
    heat(ax[2], m * (W0 != 0), "m : masque appris, ∈ [0, 1]", cmap=SEQ, vmax=1, sub="appris : atténuer ou couper")
    ax[3].axis("off"); ax[3].text(0.5, 0.5, "=", fontsize=28, ha="center", va="center", color=INK2)
    heat(ax[4], m * W0, "W = m ⊙ W₀ : réservoir sculpté", vmax=v, sub="mêmes signes, jamais plus fort")
    fig.suptitle("Partie 2 — le réservoir sculpté : on n'apprend qu'une atténuation par connexion",
                 fontsize=13, weight="bold", y=1.04)
    save(fig, "schema_sculpte.png")
    # contrôles : élagué et repondéré
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6))
    heat(ax[0], m * W0, "sculpté : m ∈ [0, 1]", vmax=v, sub="le meilleur")
    heat(ax[1], binary * W0, "élagué : m ∈ {0, 1}", vmax=v, sub="couper sans nuance : moins bien")
    heat(ax[2], g * W0, "repondéré : g libre", vmax=np.abs(g * W0).max(), sub="peut inverser, amplifier : moins bien")
    fig.suptitle("Ce qui compte : atténuer, sans pouvoir inverser ni amplifier (rouge = positif, bleu = négatif)",
                 fontsize=11.5, weight="bold", y=1.04)
    save(fig, "schema_sculpte_controles.png")


# ---------------------------------------------------------------- 4. réservoir accordé (géométrie)
def ellipse_of(M, k=200):
    t = np.linspace(0, 2 * np.pi, k)
    return M @ np.vstack([np.cos(t), np.sin(t)])


def rot(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def plane(ax, title, sub=None):
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal")
    ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=10.5, weight="bold")
    if sub:
        ax.text(0, -1.55, sub, ha="center", va="bottom", fontsize=8.5, color=INK2)


def fig_accorde():
    U, V = rot(0.6), rot(-0.3)
    sig = np.array([1.2, 0.45])
    W0 = U @ np.diag(sig) @ V.T
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.8))
    circ = ellipse_of(np.eye(2))
    plane(ax[0], "1. un état x (cercle)", "tous les états de norme 1")
    ax[0].plot(*circ, color=INK2, lw=2)
    plane(ax[1], "2. Vᵀ : tourner", "directions d'entrée du réservoir\n(figé)")
    ax[1].plot(*circ, color=FROZEN, lw=2)
    for k in range(2):
        ax[1].annotate("", xy=V[:, k] * 1.0, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=FROZEN, lw=2))
    plane(ax[2], "3. diag(σ ⊙ g) : étirer", "σ figé, g appris :\nl'amplitude de chaque direction")
    E0 = ellipse_of(np.diag(sig)); E1 = ellipse_of(np.diag(sig * [1.2, 0.35]))
    ax[2].plot(*E0, color=FROZEN, lw=2, ls="--", label="σ (réservoir)")
    ax[2].plot(*E1, color=BLUE, lw=2.5, label="σ ⊙ g (accordé)")
    ax[2].legend(loc="upper center", fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.0))
    plane(ax[3], "4. U : tourner", "directions de sortie du réservoir\n(figé)")
    ax[3].plot(*(U @ E0), color=FROZEN, lw=2, ls="--"); ax[3].plot(*(U @ E1), color=BLUE, lw=2.5)
    for k in range(2):
        ax[3].annotate("", xy=U[:, k] * 1.4, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=FROZEN, lw=1.5))
    # 5. comparaison : accordé vs tourné
    Q = rot(1.0)
    plane(ax[4], "accordé ou tourné ?", "accordé : mêmes axes, autres longueurs\ntourné : mêmes longueurs, autres axes")
    ax[4].plot(*(U @ E0), color=FROZEN, lw=2, ls="--", label="W₀")
    ax[4].plot(*(U @ E1), color=BLUE, lw=2.5, label="accordé")
    ax[4].plot(*(Q @ U @ E0), color=ORANGE, lw=2.5, label="tourné")
    ax[4].legend(loc="upper center", fontsize=8, frameon=False, ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Partie 4 — le réservoir accordé : W = U · diag(σ ⊙ g) · Vᵀ, on n'apprend que g (une amplitude par "
                 "direction)", fontsize=13, weight="bold", y=1.04)
    save(fig, "schema_accorde.png")


# ---------------------------------------------------------------- 5. autres décompositions
def fig_decompositions():
    fig = plt.figure(figsize=(16, 7.6))
    gs = fig.add_gridspec(2, 5, height_ratios=[1, 1.05], hspace=0.45)
    # Iwasawa : cercle -> N (cisaillement) -> A (étirer) -> K (tourner)
    Nm = np.array([[1, 0.9], [0, 1]]); A = np.diag([1.1, 0.55]); K = rot(0.7)
    steps = [(np.eye(2), "1. un état x", None, INK2),
             (Nm, "2. N : cisailler", "triangulaire, diagonale 1\n(cisaillement S appris)", ORANGE),
             (A @ Nm, "3. A : étirer", "amplitudes a ⊙ g\n(appris)", BLUE),
             (K @ A @ Nm, "4. K : tourner", "orthogonale Q (figé)", FROZEN)]
    for i, (M, t, sub, c) in enumerate(steps):
        ax = fig.add_subplot(gs[0, i]); plane(ax, t, sub)
        if i:
            ax.plot(*ellipse_of(steps[i - 1][0]), color=GRID, lw=1.5, ls="--")
        ax.plot(*ellipse_of(M), color=c, lw=2.5)
        # un carré pour voir le cisaillement
        sq = M @ np.array([[0, 0.7, 0.7, 0, 0], [0, 0, 0.7, 0.7, 0]])
        ax.fill(*sq, color=c, alpha=0.12, lw=0)
    ax = fig.add_subplot(gs[0, 4]); blank(ax)
    ax.text(0.0, 0.85, "Iwasawa (K · A · N, c'est-à-dire QR)", fontsize=11, weight="bold")
    ax.text(0.0, 0.05, "W₀ = Q · diag(a) · N₀\n\nL'accordé apprend des amplitudes\ndans la base des valeurs singulières ;"
            "\nIwasawa sépare en plus un\n**cisaillement**, que l'accordé\nne peut pas apprendre.".replace("**", ""),
            fontsize=9.5, color=INK2, va="bottom")
    # Jordan-Chevalley : même spectre, normal vs non normal
    ax = fig.add_subplot(gs[1, 0:3])
    lam = np.diag([0.9, 0.8])
    for c, a, lab in [(FROZEN, 0.0, "partie semi-simple seule (normale) : décroît tout de suite"),
                      (BLUE, 2.5, "+ partie nilpotente : amplification transitoire, puis décroît"),
                      (ORANGE, 6.0, "+ partie nilpotente plus forte")]:
        M = lam + np.array([[0, a], [0, 0]])
        x = np.array([0.0, 1.0]); nrm = []
        for t in range(40):
            nrm.append(np.linalg.norm(x)); x = M @ x
        ax.plot(nrm, color=c, lw=2.5, label=lab)
    ax.set_xlabel("pas de temps t"); ax.set_ylabel("‖x(t)‖, en partant de ‖x(0)‖ = 1")
    ax.set_title("Jordan-Chevalley : W = S + N. Mêmes valeurs propres (0.9 et 0.8), dynamiques différentes",
                 fontsize=11, weight="bold")
    ax.legend(frameon=False, fontsize=9); ax.grid(color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax = fig.add_subplot(gs[1, 3:5]); blank(ax)
    ax.text(0.0, 0.92, "Jordan-Chevalley (semi-simple + nilpotent)", fontsize=11, weight="bold")
    ax.text(0.0, 0.0, "S : modes propres (module, phase) — appris,\n    c'est l'« accordé sur les valeurs propres »\n"
            "N : nilpotente (Nᵏ = 0) — apprise en option\n\nLa partie nilpotente ne change aucune valeur\npropre, "
            "mais elle fait passer l'énergie\nd'un mode à l'autre : un signal est amplifié\nquelques pas avant de "
            "s'éteindre.\nC'est la non-normalité, liée à la mémoire\n(Ganguli et al., 2008).", fontsize=9.5,
            color=INK2, va="bottom")
    fig.suptitle("Parties 5 et 6 — deux autres façons d'écrire la matrice du réservoir", fontsize=13, weight="bold",
                 y=0.98)
    save(fig, "schema_decompositions.png")


# ---------------------------------------------------------------- 6. couplages
def coupled_panel(ax, title, top, bottom, down, up, readout, note):
    """top, bottom : (texte, appris?, sous-titre, boucle sur soi?) ;
    down : flèche du bloc du haut vers celui du bas, (retard?, appris?) ; up : du bas vers le haut ;
    readout : "top" ou "bottom"."""
    blank(ax, (0, 3), (0, 2.6))
    ax.text(1.5, 2.55, title, ha="center", va="top", fontsize=11.5, weight="bold")
    box(ax, (0.95, 1.55), 1.1, 0.55, top[0], top[1], top[2])
    box(ax, (0.95, 0.35), 1.1, 0.55, bottom[0], bottom[1], bottom[2])
    ax.text(0.15, 1.2, "u(t)", fontsize=10, ha="center", va="center")
    arrow(ax, (0.32, 1.28), (0.93, 1.75), None); arrow(ax, (0.32, 1.12), (0.93, 0.62), None)
    if down:
        arrow(ax, (1.75, 1.53), (1.75, 0.92), down[1], "z⁻¹" if down[0] else "", lab_off=(0.2, 0))
    if up:
        arrow(ax, (1.25, 0.92), (1.25, 1.53), up[1], "z⁻¹" if up[0] else "", lab_off=(-0.2, 0))
    for blk, Y in ((top, 1.55), (bottom, 0.35)):
        if blk[3]:
            c = BLUE if blk[1] else FROZEN
            ax.add_patch(FancyArrowPatch((2.07, Y + 0.4), (2.07, Y + 0.16), arrowstyle="-|>", mutation_scale=12,
                                         lw=1.6, color=c, connectionstyle="arc3,rad=-2.5"))
    Y = 1.83 if readout == "top" else 0.62
    arrow(ax, (2.38, Y), (2.9, Y), True, None); ax.text(2.95, Y, "ŷ", fontsize=11, ha="left", va="center")
    ax.text(1.5, 0.05, note, ha="center", va="bottom", fontsize=8.5, color=INK2)


def fig_couplage():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2))
    R = ("réservoir y", False, "W_R figé", True)
    A = ("RNN x", True, "W_A appris", True)
    coupled_panel(axes[0, 0], "Forme 1 : duo croisé, lu côté RNN", R, A, (True, True), (True, True), "bottom",
                  "deux états à mémoire qui se lisent\navec un pas de retard ; readout sur x")
    coupled_panel(axes[0, 1], "Forme 2 : duo croisé, lu côté réservoir", R, A, (True, True), (True, True), "top",
                  "mêmes équations, readout sur y\n(résultat : jamais mieux que l'accordé)")
    coupled_panel(axes[0, 2], "Forme 3 : relais aléatoire", ("relais y", False, "figé, sans mémoire", False),
                  ("RNN x", True, "W_A appris", False), (True, True), (False, False), "bottom",
                  "y(t) = tanh(W_R x(t) + …) figé\nx(t+1) = tanh(W_A y(t) + …) : gagne sur Hénon")
    coupled_panel(axes[1, 0], "Forme 4 : relais appris", ("réservoir x", False, "W_R figé", False),
                  ("relais y", True, "appris, sans mémoire", False), (False, True), (True, False), "top",
                  "y(t) = tanh(W_A x(t) + …) appris\nx(t+1) = tanh(W_R y(t) + …) figé ; readout sur x")
    coupled_panel(axes[1, 1], "Contrôle : relais tout appris (DT-RNN)", ("relais y", True, "appris, sans mémoire", False),
                  ("RNN x", True, "W_A appris", False), (True, True), (False, True), "bottom",
                  "la forme 3, mais tout appris : aussi bonne\n→ c'est la profondeur de la transition qui compte")
    ax = axes[1, 2]; blank(ax, (0, 3), (0, 2.6))
    ax.text(1.5, 2.55, "Contrôle : readout carré", ha="center", va="top", fontsize=11.5, weight="bold")
    box(ax, (0.2, 1.0), 1.0, 0.6, "accordé x", True)
    arrow(ax, (1.22, 1.42), (1.85, 1.75), None, "x"); arrow(ax, (1.22, 1.18), (1.85, 0.85), None, "x²")
    ax.text(2.0, 1.3, "[x, x²]", fontsize=12, va="center")
    arrow(ax, (2.6, 1.3), (2.9, 1.3), True, None); ax.text(2.95, 1.3, "ŷ", fontsize=11, va="center")
    ax.text(1.5, 0.05, "aucun couplage : les produits de variables\narrivent par le readout (≈ NVAR)",
            ha="center", va="bottom", fontsize=8.5, color=INK2)
    legend_frozen_learned(fig, y=0.03, extra=[(INK2, "z⁻¹ : lu avec un pas de retard")])
    fig.suptitle("Partie 7 — coupler un réservoir figé et un RNN appris : quatre formes et deux contrôles",
                 fontsize=13, weight="bold", y=0.97)
    save(fig, "schema_couplage.png")


# ---------------------------------------------------------------- 7. améliorations : portes, linéaire, rapide
def fig_ameliorations():
    fig = plt.figure(figsize=(16, 4.0))
    gs = fig.add_gridspec(1, 3, wspace=0.25)
    # portes
    ax = fig.add_subplot(gs[0, 0]); blank(ax, (0, 3), (0, 2.4))
    ax.text(1.5, 2.35, "Accordé à portes", ha="center", va="top", fontsize=11.5, weight="bold")
    box(ax, (0.2, 1.35), 1.1, 0.5, "porte z", True, "U·diag(σ⊙g_z)·Vᵀ")
    box(ax, (1.7, 1.35), 1.1, 0.5, "candidat h̃", True, "U·diag(σ⊙g_h)·Vᵀ")
    box(ax, (0.95, 0.35), 1.1, 0.5, "x(t+1)", None if False else True, "(1−z)·x + z·h̃")
    arrow(ax, (0.75, 1.33), (1.2, 0.87), True); arrow(ax, (2.25, 1.33), (1.8, 0.87), True)
    ax.text(1.5, 0.05, "chaque matrice de la porte garde les directions\nfigées du réservoir ; seul un profil "
            "d'amplitudes apprend", ha="center", va="bottom", fontsize=8.5, color=INK2)
    # LRU accordé : valeurs propres dans le disque
    ax = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(2)
    r = np.sqrt(rng.uniform(0.5 ** 2, 0.99 ** 2, 14)); th = rng.uniform(0, np.pi, 14)
    lam0 = r * np.exp(1j * th)
    r1 = np.clip(r * rng.uniform(0.85, 1.12, 14), 0, 0.995); th1 = th + rng.normal(0, 0.18, 14)
    lam1 = r1 * np.exp(1j * th1)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color=GRID, lw=1.5)
    for a, b in zip(lam0, lam1):
        ax.annotate("", xy=(b.real, b.imag), xytext=(a.real, a.imag),
                    arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2))
    ax.plot(lam0.real, lam0.imag, "o", color=FROZEN, ms=6, label="valeurs propres de départ")
    ax.plot(lam1.real, lam1.imag, "o", color=BLUE, ms=6, label="après apprentissage")
    ax.set_aspect("equal"); ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.25, 1.15)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.axhline(0, color=GRID, lw=1)
    ax.set_title("Accordé linéaire (type LRU)", fontsize=11.5, weight="bold")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.text(0, -0.22, "h(t) = λ ⊙ h(t−1) + B u(t) : seules λ (module, phase)\net B apprennent ; les sorties C restent "
            "aléatoires", ha="center", va="top", fontsize=8.5, color=INK2)
    # accordé rapide : Fourier
    ax = fig.add_subplot(gs[0, 2]); blank(ax, (0, 3), (0, 2.4))
    ax.text(1.5, 2.35, "Accordé rapide (jusqu'à 16 000 neurones)", ha="center", va="top", fontsize=11.5,
            weight="bold")
    labels = [("P₂", False, 0.36), ("F", False, 0.36), ("λ ⊙ g", True, 0.6), ("F⁻¹", False, 0.4), ("P₁", False, 0.36)]
    x0 = 0.12
    for i, (t_, l_, w_) in enumerate(labels):
        if i:
            arrow(ax, (x0 - 0.16, 1.45), (x0, 1.45), None, lw=1.4)
        box(ax, (x0, 1.2), w_, 0.5, t_, l_)
        x0 += w_ + 0.18
    ax.text(1.5, 0.45, "permutations et transformée de Fourier : figées\nseules les amplitudes g apprennent\n"
            "coût N·log N par pas au lieu de N²", ha="center", va="center", fontsize=9, color=INK2)
    fig.suptitle("Partie 4 — étendre l'accordé : des portes, une récurrence linéaire, un réservoir rapide", fontsize=13,
                 weight="bold", y=1.04)
    legend_frozen_learned(fig, y=-0.04)
    save(fig, "schema_ameliorations.png")


# ---------------------------------------------------------------- 8. combien on apprend (vue d'ensemble)
def fig_budget():
    # nombres de paramètres appris de la partie récurrente à 500 neurones (une entrée), tirés des résultats
    models = [("Réservoir classique", 0, FROZEN), ("Accordé linéaire (type LRU)", 1_250, BLUE),
              ("Réservoir accordé", 1_500, BLUE), ("Accordé sélectif", 2_000, BLUE),
              ("Accordé à portes", 4_500, BLUE), ("Réservoir corrigé de rang 4", 5_500, ORANGE),
              ("Réservoir sculpté intégral", 250_500, ORANGE), ("RNN classique", 251_000, INK2),
              ("LRU", 251_750, INK2), ("GRU", 754_500, INK2), ("LSTM", 1_006_000, INK2)]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    y = np.arange(len(models))[::-1]
    for yi, (n, p, c) in zip(y, models):
        ax.plot([1, max(p, 1)], [yi, yi], color=GRID, lw=2)
        ax.plot(max(p, 1), yi, "o", color=c, ms=9)
        ax.text(max(p, 1) * 1.35, yi, f"{p:,}".replace(",", " "), va="center", fontsize=9, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([m[0] for m in models])
    ax.set_xscale("log"); ax.set_xlim(0.8, 4e6)
    ax.set_xlabel("paramètres appris dans la partie récurrente, 500 neurones (échelle log ; le réservoir classique n'en a aucun)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_title("Combien chaque modèle apprend : trois ordres de grandeur d'écart", fontsize=12, weight="bold")
    fig.text(0.13, -0.04, "■", color=BLUE, fontsize=12); fig.text(0.145, -0.04, "accordé : quelques milliers", fontsize=9, color=INK2)
    fig.text(0.36, -0.04, "■", color=ORANGE, fontsize=12); fig.text(0.375, -0.04, "autres modèles près du réservoir", fontsize=9, color=INK2)
    fig.text(0.64, -0.04, "■", color=INK2, fontsize=12); fig.text(0.655, -0.04, "modèles existants (tout appris)", fontsize=9, color=INK2)
    save(fig, "schema_budget.png")


# ---------------------------------------------------------------- 9. la frise : du réservoir au RNN
def fig_continuum():
    """Chaque famille placée selon le nombre de paramètres récurrents appris (100 neurones) ; la couleur dit ce qui
    contraint l'apprentissage."""
    n = 100
    fams = [  # (partie, nom, paramètres récurrents appris à N = 100, ligne, ce qui est appris / la contrainte)
        (0, "Réservoir classique", 0.6, 0, "rien : W = W₀"),
        (4, "Réservoir accordé", n, 1, "N amplitudes : W = U·diag(σ⊙g)·Vᵀ"),
        (1, "Réservoir corrigé de rang 4", 2 * 4 * n, 2, "W = W₀ + U·Vᵀ, rang 4"),
        (3, "Réservoir tourné", n * (n - 1) // 2, 3, "une rotation : W = Q·W₀·Qᵀ (spectre figé)"),
        (5, "Réservoir cisaillé", n * (n - 1) // 2, 4, "un cisaillement triangulaire (Iwasawa)"),
        (6, "Réservoir nilpotent", n * (n - 1) // 2 + 2 * n, 5, "modes propres + partie nilpotente"),
        (2, "Réservoir sculpté", n * n, 6, "une atténuation m ∈ [0, 1] par connexion"),
        (1, "RNN freiné, mélange", n * n, 7, "tout, mais la récurrence 10 fois plus lentement"),
        (0, "RNN classique", n * n, 8, "tout, librement"),
    ]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    cmap = {0: INK2, 1: ORANGE, 2: ORANGE, 3: AQUA, 4: BLUE, 5: AQUA, 6: AQUA}
    for part, name, p, row, what in fams:
        y = 8 - row
        c = FROZEN if p < 1 else cmap[part]
        ax.plot([0.6, p], [y, y], color=GRID, lw=2, zorder=1)
        ax.plot(p, y, "o", ms=12, color=c, zorder=3)
        lab = f"Partie {part} · {name}" if part else name
        ax.text(p * 1.35, y + 0.13, lab, fontsize=10.5, weight="bold", va="center")
        ax.text(p * 1.35, y - 0.25, what, fontsize=8.8, color=INK2, va="center")
    ax.set_xscale("log"); ax.set_xlim(0.45, 2e6); ax.set_ylim(-0.9, 8.8)
    ax.set_yticks([]); ax.set_xticks([1, 10, 100, 1_000, 10_000])
    ax.set_xticklabels(["0", "10", "100", "1 000", "10 000 (= N²)"])
    ax.set_xlabel("paramètres récurrents appris, réseau de 100 neurones (échelle log ; W_in et les biais en plus)")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.annotate("", xy=(1.3e4, -0.65), xytext=(0.6, -0.65),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.5))
    ax.text(0.65, -0.45, "réservoir : la dynamique reste celle du hasard", fontsize=8.5, color=INK2)
    ax.text(1.2e4, -0.45, "RNN : la dynamique est entièrement apprise", fontsize=8.5, color=INK2, ha="right")
    ax.set_title("Entre un réservoir et un RNN : chaque partie choisit ce que la récurrence a le droit d'apprendre\n"
                 "(la partie 7 garde au contraire deux réseaux distincts, un réservoir figé et un RNN, et les couple)",
                 fontsize=12, weight="bold")
    save(fig, "schema_continuum.png")


# ---------------------------------------------------------------- 10. réservoir tourné
def fig_tourne():
    U, V = rot(0.6), rot(-0.3)
    sig = np.array([1.2, 0.45])
    W0 = U @ np.diag(sig) @ V.T
    Q = rot(1.0)
    W = Q @ W0 @ Q.T
    fig = plt.figure(figsize=(15, 4.2))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.25])
    ax = fig.add_subplot(gs[0]); plane(ax, "1. le réservoir W₀", "un cercle d'états devient une ellipse")
    ax.plot(*ellipse_of(np.eye(2)), color=GRID, lw=1.5, ls="--"); ax.plot(*ellipse_of(W0), color=FROZEN, lw=2.5)
    ax = fig.add_subplot(gs[1]); plane(ax, "2. Q : une rotation apprise", "Q = exp(S − Sᵀ), départ Q = I")
    for k in range(2):
        ax.annotate("", xy=Q[:, k], xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2))
        ax.annotate("", xy=np.eye(2)[:, k], xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=FROZEN, lw=1.2))
    ax = fig.add_subplot(gs[2]); plane(ax, "3. W = Q · W₀ · Qᵀ", "même ellipse, tournée :\nmêmes longueurs, autres axes")
    ax.plot(*ellipse_of(W0), color=FROZEN, lw=2, ls="--", label="W₀")
    ax.plot(*ellipse_of(W), color=BLUE, lw=2.5, label="tourné")
    ax.legend(loc="upper center", fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(0.5, 1.0))
    # valeurs propres : identiques
    ax = fig.add_subplot(gs[3])
    rng = np.random.default_rng(3)
    A = rng.normal(size=(40, 40)) / np.sqrt(40) * 0.9
    Qb, _ = np.linalg.qr(rng.normal(size=(40, 40)))
    l0, l1 = np.linalg.eigvals(A), np.linalg.eigvals(Qb @ A @ Qb.T)
    t = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(t), np.sin(t), color=GRID, lw=1)
    ax.plot(l0.real, l0.imag, "o", ms=8, mfc="none", mec=FROZEN, mew=1.5, label="valeurs propres de W₀")
    ax.plot(l1.real, l1.imag, ".", ms=6, color=BLUE, label="valeurs propres de Q·W₀·Qᵀ")
    ax.set_aspect("equal"); ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.2))
    ax.set_title("4. le spectre ne bouge pas (40 neurones)", fontsize=10.5, weight="bold")
    fig.suptitle("Partie 3 — le réservoir tourné : on n'apprend que l'orientation de la dynamique, jamais ses amplitudes",
                 fontsize=13, weight="bold", y=1.05)
    save(fig, "schema_tourne.png")


# ---------------------------------------------------------------- 11. réservoir cisaillé
def tri_icon(ax, n, band, title, c, unit_diag=True):
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if band is None or j - i <= band:
                M[i, j] = 1
    ax.imshow(np.where(np.eye(n) > 0, 0.35 if unit_diag else 0, M), cmap=LinearSegmentedColormap.from_list("t", [SURF, c]), vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(GRID)
    k = int(M.sum())
    ax.set_title(title, fontsize=10, weight="bold")
    ax.text(n / 2 - 0.5, n + 0.6, f"{k} coefficients appris\n(sur {n} neurones)", ha="center", va="top", fontsize=8.5,
            color=INK2)


def fig_cisaille():
    fig = plt.figure(figsize=(16, 4.4))
    gs = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 0.9, 0.9], wspace=0.35)
    Nm = np.array([[1, 0.9], [0, 1]]); A = np.diag([1.1, 0.55]); K = rot(0.7)
    steps = [(np.eye(2), "1. un état x", None, INK2),
             (Nm, "2. N : cisailler", "triangulaire, diagonale 1\n(le cisaillement S est appris)", BLUE),
             (A @ Nm, "3. A : étirer", "amplitudes a (figées)", FROZEN),
             (K @ A @ Nm, "4. K : tourner", "orthogonale Q (figée)", FROZEN)]
    for i, (M, t, sub, c) in enumerate(steps):
        ax = fig.add_subplot(gs[i]); plane(ax, t, sub)
        if i:
            ax.plot(*ellipse_of(steps[i - 1][0]), color=GRID, lw=1.5, ls="--")
        ax.plot(*ellipse_of(M), color=c, lw=2.5)
        sq = M @ np.array([[0, 0.7, 0.7, 0, 0], [0, 0, 0.7, 0.7, 0]])
        ax.fill(*sq, color=c, alpha=0.15, lw=0)
    tri_icon(fig.add_subplot(gs[4]), 12, None, "cisaillement plein", BLUE)
    tri_icon(fig.add_subplot(gs[5]), 12, 3, "en bande (k = 3)", BLUE)
    fig.suptitle("Partie 5 — le réservoir cisaillé : W₀ = Q·diag(a)·N₀ (Iwasawa, c'est-à-dire QR) ; "
                 "on n'apprend que la partie triangulaire, qui rend la matrice non normale", fontsize=12.5,
                 weight="bold", y=0.98)
    save(fig, "schema_cisaille.png")


# ---------------------------------------------------------------- 12. réservoir nilpotent
def fig_nilpotent():
    fig = plt.figure(figsize=(15, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 0.8, 1.1], wspace=0.3)
    ax = fig.add_subplot(gs[0])
    lam = np.diag([0.9, 0.8])
    for c, a, lab in [(FROZEN, 0.0, "partie semi-simple seule (normale) : décroît tout de suite"),
                      (BLUE, 2.5, "+ partie nilpotente : amplification transitoire, puis décroît"),
                      (ORANGE, 6.0, "+ partie nilpotente plus forte")]:
        M = lam + np.array([[0, a], [0, 0]])
        x = np.array([0.0, 1.0]); nrm = []
        for _ in range(40):
            nrm.append(np.linalg.norm(x)); x = M @ x
        ax.plot(nrm, color=c, lw=2.5, label=lab)
    ax.set_xlabel("pas de temps t"); ax.set_ylabel("‖x(t)‖, en partant de ‖x(0)‖ = 1")
    ax.set_title("Mêmes valeurs propres (0.9 et 0.8), dynamiques différentes", fontsize=11, weight="bold")
    ax.legend(frameon=False, fontsize=8.5); ax.grid(color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    tri_icon(fig.add_subplot(gs[1]), 12, None, "T, strictement triangulaire", ORANGE, unit_diag=False)
    ax = fig.add_subplot(gs[2]); blank(ax)
    ax.text(0.0, 0.95, "W = S(g, φ) + Qr · T · Qrᵀ", fontsize=12, weight="bold", va="top")
    ax.text(0.0, 0.05, "S : chaque mode propre de W₀, dont on apprend\n    le module g et la phase φ (bleu)\n"
            "T : triangulaire stricte apprise, donc Tᴺ = 0 :\n    Qr·T·Qrᵀ est nilpotente (orange)\n"
            "Qr : rotation aléatoire figée (gris)\n\nLa partie nilpotente ne change aucune valeur\n"
            "propre mais fait passer l'énergie d'un mode\nà l'autre (non-normalité).", fontsize=9.5, color=INK2,
            va="bottom")
    fig.suptitle("Partie 6 — le réservoir nilpotent : la décomposition de Jordan-Chevalley, semi-simple + nilpotente",
                 fontsize=13, weight="bold", y=1.03)
    save(fig, "schema_nilpotent.png")


if __name__ == "__main__":
    fig_principe(); fig_partie1(); fig_sculpte(); fig_accorde(); fig_decompositions(); fig_couplage()
    fig_ameliorations(); fig_budget(); fig_continuum(); fig_tourne(); fig_cisaille(); fig_nilpotent()
