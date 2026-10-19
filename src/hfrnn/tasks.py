"""Tâches : Mackey-Glass (+15 pas), NARMA10, mémoire pure (délai 15), Fibonacci mod 10 (démo),
et trois séries réelles (partie 8), téléchargées au premier usage dans data/ (non redistribuées) :

    ETTh1 (+24 h)              température d'huile d'un transformateur électrique, horaire, 17 420 pas
                               (Zhou et al., 2021, github.com/zhouhaoyi/ETDataset, licence CC BY-ND 4.0)
    Températures Melbourne     température minimale quotidienne, 1981-1990, 3 650 pas (Bureau of Meteorology)
    Taches solaires            nombre mensuel de taches solaires, 1749-1983, 2 820 pas (SIDC)

Chaque tâche renvoie (u, y) : l'entrée et la cible, une valeur par pas de temps.
Pour les séries réelles : u(t) = valeur à t, y(t) = variation s(t + horizon) − s(t) (prédire « ce qui change »,
comme en prévision ; prédire une variation nulle revient à la prévision naïve « demain = aujourd'hui »).
75 % de la série pour l'entraînement, 25 % pour le test.
"""
import os
import urllib.request
from pathlib import Path

import numpy as np


def task_fib(T):
    f = [0, 1]
    for _ in range(T + 1):
        f.append((f[-1] + f[-2]) % 10)
    f = np.array(f, float)
    return f[:T], f[1:T + 1]


def task_mackey_glass(T, horizon=15):
    dt, delay, sub = 0.1, 170, 20            # tau = 17, échantillonné tous les 2 temps
    n = (T + horizon + 200) * sub
    x = np.full(n + delay, 1.2)
    for i in range(delay, n + delay - 1):
        xd = x[i - delay]
        x[i + 1] = x[i] + dt * (0.2 * xd / (1 + xd ** 10) - 0.1 * x[i])
    s = x[delay::sub][200:][:T + horizon]
    return s[:T], s[horizon:T + horizon]


def task_narma10(T, seed=123):
    r = np.random.default_rng(seed)
    u = r.uniform(0, 0.5, T + 1)
    y = np.zeros(T + 1)
    for t in range(9, T):
        y[t + 1] = (0.3 * y[t] + 0.05 * y[t] * y[t - 9:t + 1].sum()
                    + 1.5 * u[t - 9] * u[t] + 0.1)
    assert np.all(np.isfinite(y))
    return u[:T], y[1:T + 1]


def task_delay(T, delay=15, seed=11):
    """Mémoire pure : rappeler l'entrée d'il y a `delay` pas (entrée i.i.d. uniforme)."""
    r = np.random.default_rng(seed)
    u = r.uniform(-1, 1, T + delay)
    return u[delay:], u[:T]


DATA_DIR = Path(os.environ.get("HFRNN_DATA_DIR", Path(__file__).resolve().parents[2] / "data"))
SOURCES = {
    "ETTh1.csv": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
    "daily-min-temperatures.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv",
    "monthly-sunspots.csv": "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-sunspots.csv",
}


def read_column(fname, col):
    """Colonne numérique d'un CSV téléchargé (et gardé en cache dans data/)."""
    path = DATA_DIR / fname
    if not path.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(SOURCES[fname], path)
    lines = path.read_text().strip().splitlines()
    head = [h.strip('"') for h in lines[0].split(",")]
    j = head.index(col)
    return np.array([float(l.split(",")[j].strip('"')) for l in lines[1:]])


def real_task(fname, col, horizon):
    def fn(T=None):                                    # T ignoré : on prend toute la série
        s = read_column(fname, col)
        return s[:-horizon], s[horizon:] - s[:-horizon]
    return fn


TASKS = {
    "Mackey-Glass (+15)": task_mackey_glass,      # prédiction d'une série chaotique, 15 pas à l'avance
    "NARMA10": task_narma10,                      # système non linéaire avec mémoire d'ordre 10
    "Mémoire (délai 15)": task_delay,             # mémoire pure : rappeler l'entrée d'il y a 15 pas
    "Fibonacci mod 10": task_fib,                 # démo uniquement
    # partie 8 : données réelles
    "ETTh1, variation à 24 h": real_task("ETTh1.csv", "OT", 24),
    "Melbourne, variation à 1 jour": real_task("daily-min-temperatures.csv", "Temp", 1),
    "Taches solaires, variation à 1 mois": real_task("monthly-sunspots.csv", "Sunspots", 1),
}
REAL = ["ETTh1, variation à 24 h", "Melbourne, variation à 1 jour", "Taches solaires, variation à 1 mois"]


def load(name, T=None):
    """Série normalisée (statistiques de la partie entraînement)."""
    from .core import T as T_DEFAULT, n_train
    T = T_DEFAULT if T is None else T
    fn = TASKS[name]
    u, y = fn(T)
    nt = n_train(u)
    u = (u - u[:nt].mean()) / u[:nt].std()
    y = (y - y[:nt].mean()) / y[:nt].std()
    return u, y
