"""
Démo d'un Echo State Network (reservoir computing) en numpy.

Entrée : suite de Fibonacci modulo 10 (0,1,1,2,3,5,8,3,1,4,5,9,4,3,7,0,7,7,4,1,5,6,1,7,8,5,3,8,1,9,0,9,9,8,7,5,2,7,9,6,5,1,6,7,3,0,3,3,6,9,5,4,9,3,2,5,7,2,9,1 ...)
La suite de Fibonacci brute explose, donc on la prend modulo 10 : elle devient
périodique (période 60) mais reste "non triviale" : le terme suivant dépend
des DEUX termes précédents, donc le réservoir doit avoir de la mémoire.

Tâche : à chaque instant t, prédire u(t+1) à partir de u(t).
Seul le readout (W_out) est entraîné, par régression ridge.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from hfrnn.core import out_path

rng = np.random.default_rng(42)

# ----------------------------------------------------------------------
# 1. Données : Fibonacci mod 10
# ----------------------------------------------------------------------
T = 800
fib = [0, 1]
for _ in range(T + 1):
    fib.append((fib[-1] + fib[-2]) % 10)
fib = np.array(fib, dtype=float)

u = fib[:T] / 9.0          # entrée normalisée dans [0, 1]
y = fib[1:T + 1] / 9.0     # cible : valeur suivante

# ----------------------------------------------------------------------
# 2. Réservoir (FIXE, aléatoire)
# ----------------------------------------------------------------------
N = 100              # nombre de neurones
sparsity = 0.10      # 10 % de connexions non nulles
spectral_radius = 0.9
input_scale = 1.0
leak = 1.0           # 1.0 = pas de fuite
ridge = 1e-6

W_in = rng.uniform(-input_scale, input_scale, size=(N, 1))

W = rng.uniform(-1, 1, size=(N, N))
W *= rng.random((N, N)) < sparsity                     # matrice creuse
W *= spectral_radius / max(abs(np.linalg.eigvals(W)))  # rayon spectral voulu

# ----------------------------------------------------------------------
# 3. On fait tourner le réservoir et on enregistre tous les états
# ----------------------------------------------------------------------
X = np.zeros((T, N))
x = np.zeros(N)
for t in range(T):
    x_new = np.tanh(W_in[:, 0] * u[t] + W @ x)
    x = (1 - leak) * x + leak * x_new
    X[t] = x

# ----------------------------------------------------------------------
# 4. Entraînement du readout (seule partie apprise)
# ----------------------------------------------------------------------
washout = 100          # on ignore le début (états initiaux pas encore "oubliés")
n_train = 600

# on ajoute un biais (colonne de 1)
Xb = np.hstack([X, np.ones((T, 1))])
A, b = Xb[washout:n_train], y[washout:n_train]
W_out = np.linalg.solve(A.T @ A + ridge * np.eye(N + 1), A.T @ b)

pred = Xb @ W_out
pred_digits = np.clip(np.round(pred * 9), 0, 9)   # on arrondit vers un chiffre 0..9
true_digits = y * 9

test = slice(n_train, T)
mse = np.mean((pred[test] - y[test]) ** 2)
acc = np.mean(pred_digits[test] == true_digits[test])
print(f"MSE test : {mse:.5f}")
print(f"Chiffres exacts sur le test : {acc * 100:.1f} %")

# ----------------------------------------------------------------------
# 5. Visualisations
# ----------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Echo State Network sur Fibonacci mod 10", fontsize=15)

# (a) Le réservoir vu comme un graphe
ax = axes[0, 0]
angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
pos = np.c_[np.cos(angles), np.sin(angles)]
src, dst = np.nonzero(W.T)          # W[i, j] : j -> i
segs = [[pos[s], pos[d]] for s, d in zip(src, dst)]
colors = ["tab:red" if W[d, s] > 0 else "tab:blue" for s, d in zip(src, dst)]
widths = [0.2 + 1.5 * abs(W[d, s]) for s, d in zip(src, dst)]
ax.add_collection(LineCollection(segs, colors=colors, linewidths=widths, alpha=0.35))
ax.scatter(pos[:, 0], pos[:, 1], s=25, c="black", zorder=3)
ax.scatter([-1.6], [0], s=200, c="tab:green", marker="s", zorder=3)
ax.annotate("entrée u(t)", (-1.6, 0), (-1.6, 0.18), ha="center")
for i in range(0, N, 4):   # quelques flèches entrée -> neurones
    ax.plot([-1.6, pos[i, 0]], [0, pos[i, 1]], color="tab:green", alpha=0.05)
ax.set_title(f"Réservoir : {N} neurones, {len(src)} connexions\n(rouge = excitatrice, bleu = inhibitrice)")
ax.set_xlim(-1.9, 1.3); ax.set_ylim(-1.3, 1.3)
ax.set_aspect("equal"); ax.axis("off")

# (b) La matrice W en heatmap
ax = axes[0, 1]
im = ax.imshow(W, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_title("Matrice W du réservoir (fixe, creuse)\nW[i,j] = poids du neurone j vers i")
ax.set_xlabel("neurone source j"); ax.set_ylabel("neurone cible i")
plt.colorbar(im, ax=ax, fraction=0.046)

# (c) L'entrée
ax = axes[0, 2]
ax.step(range(120), u[:120] * 9, where="post")
ax.set_title("Entrée : Fibonacci mod 10 (120 premiers pas)")
ax.set_xlabel("t"); ax.set_ylabel("valeur")

# (d) Activité de tous les neurones
ax = axes[1, 0]
im = ax.imshow(X[:120].T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_title("États du réservoir x(t) : chaque ligne = un neurone")
ax.set_xlabel("t"); ax.set_ylabel("neurone")
plt.colorbar(im, ax=ax, fraction=0.046)

# (e) Quelques neurones
ax = axes[1, 1]
for i in range(5):
    ax.plot(X[:120, i], label=f"neurone {i}")
ax.set_title("5 neurones au hasard : l'entrée est\nprojetée en dynamiques variées")
ax.set_xlabel("t"); ax.legend(fontsize=8)

# (f) Prédiction vs réalité sur la zone de test
ax = axes[1, 2]
k = slice(n_train, n_train + 80)
t_axis = np.arange(n_train, n_train + 80)
ax.step(t_axis, true_digits[k], where="post", label="vrai u(t+1)", lw=2)
ax.step(t_axis, pred_digits[k], where="post", label="prédit (arrondi)", ls="--", lw=2)
ax.set_title(f"Test (jamais vu à l'entraînement)\n{acc * 100:.1f} % de chiffres exacts")
ax.set_xlabel("t"); ax.legend()

plt.tight_layout()
plt.savefig(out_path("figures", "esn_demo.png"), dpi=130)
plt.show()
