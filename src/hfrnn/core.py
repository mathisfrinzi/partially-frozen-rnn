"""Briques de base : réglages, réservoir aléatoire, ridge, BPTT à la main, Adam.

Tout est en numpy pur (pas de PyTorch). Les réglages peuvent être surchargés
par variables d'environnement, pratique pour des tests rapides :
    HFRNN_STEPS=20 HFRNN_NSEEDS=1 HFRNN_RESULTS_DIR=/tmp/res python experiments/...
"""
import os
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(os.environ.get("HFRNN_RESULTS_DIR", "results"))


def out_path(kind, name):
    """Chemin de sortie results/<kind>/<name> (kind : figures, data ou logs)."""
    d = RESULTS_DIR / kind
    d.mkdir(parents=True, exist_ok=True)
    return d / name


# ----------------------------------------------------------------------
# Réglages
# ----------------------------------------------------------------------
N = 100                 # neurones
RHO = 0.9               # rayon spectral du réservoir aléatoire
SPARSITY = 0.10
IN_SCALE = 0.5
T = 4000                # longueur de chaque série
N_TRAIN = 3000          # le reste = test
TRAIN_FRAC = N_TRAIN / T   # séries d'une autre longueur (données réelles) : même proportion


def n_train(u):
    """Nombre de pas d'entraînement : 3 000 pour les séries de 4 000 pas, 75 % sinon."""
    return int(round(TRAIN_FRAC * len(u)))
WASHOUT = 100
STEPS = int(os.environ.get("HFRNN_STEPS", 1500))   # pas d'entraînement (BPTT)
BATCH, WIN_LEN, LOSS_WASH = 32, 60, 15
LR = 2e-3
RIDGE = 1e-6
SEEDS = list(range(int(os.environ.get("HFRNN_NSEEDS", 3))))


# ----------------------------------------------------------------------
# Briques de base
# ----------------------------------------------------------------------
def make_reservoir(rng, n=None, rho=None, sparsity=None):
    """Réservoir aléatoire : W_in (n, 1) et W (n, n) de densité `sparsity` et de rayon spectral rho."""
    n = N if n is None else n
    rho = RHO if rho is None else rho
    sparsity = SPARSITY if sparsity is None else sparsity
    Win = rng.uniform(-IN_SCALE, IN_SCALE, (n, 1))
    W = rng.uniform(-1, 1, (n, n)) * (rng.random((n, n)) < sparsity)
    W *= rho / max(abs(np.linalg.eigvals(W)))
    return Win, W


def run_states(u, Win, W, b):
    """Fait tourner le réseau sur toute la série (état initial = 0)."""
    n = W.shape[0]
    X = np.zeros((len(u), n))
    x = np.zeros(n)
    for t in range(len(u)):
        x = np.tanh(Win[:, 0] * u[t] + W @ x + b)
        X[t] = x
    return X


def ridge_fit(X, y, lam=RIDGE):
    A = np.hstack([X, np.ones((len(X), 1))])
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)


def ridge_predict(X, w):
    return np.hstack([X, np.ones((len(X), 1))]) @ w


def nmse(pred, y):
    return np.mean((pred - y) ** 2) / np.var(y)


def spectral_radius(W):
    return max(abs(np.linalg.eigvals(W)))


# ----------------------------------------------------------------------
# BPTT à la main (RNN tanh, 1 entrée, 1 sortie)
# ----------------------------------------------------------------------
def bptt(U, Y, Win, W, b, Wout, c):
    """U, Y : (B, L). Renvoie loss et gradients."""
    B, L = U.shape
    n = W.shape[0]
    X = np.zeros((L + 1, B, n))
    for t in range(L):
        X[t + 1] = np.tanh(U[:, t:t + 1] @ Win.T + X[t] @ W.T + b)
    pred = X[1:] @ Wout + c                     # (L, B, 1)
    err = pred[LOSS_WASH:, :, 0] - Y.T[LOSS_WASH:]
    n_terms = err.size
    loss = np.mean(err ** 2)

    dpred = np.zeros((L, B, 1))
    dpred[LOSS_WASH:, :, 0] = 2 * err / n_terms
    dWout = np.einsum("tbn,tbo->no", X[1:], dpred)
    dc = dpred.sum(axis=(0, 1))

    dWin = np.zeros_like(Win)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)
    dx_next = np.zeros((B, n))
    for t in range(L - 1, -1, -1):
        dx = dpred[t] @ Wout.T + dx_next
        da = dx * (1 - X[t + 1] ** 2)
        dWin += da.T @ U[:, t:t + 1]
        dW += da.T @ X[t]
        db += da.sum(axis=0)
        dx_next = da @ W
    return loss, dWin, dW, db, dWout, dc


class Adam:
    """Adam ; lr peut être un nombre ou une liste (un lr par paramètre)."""

    def __init__(self, params, lr):
        self.p = params
        self.lrs = list(lr) if isinstance(lr, (list, tuple)) else [lr] * len(params)
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(self.p, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            p -= self.lrs[i] * mh / (np.sqrt(vh) + 1e-8)
