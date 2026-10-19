"""Partie 8.5 — carte des tâches : pour quelles données le réservoir accordé marche-t-il ?

Chaque générateur renvoie (u, y, n_train) : l'entrée (T,) ou (T, d), la cible (T,) et le nombre de pas
d'entraînement (le reste sert au test). Toutes les séries sont synthétiques et reproductibles (graine fixe),
normalisées (entrée et cible de variance 1 sur la partie entraînement).

Familles :
    systèmes chaotiques        Lorenz, Hénon, Lorenz-96 (40 variables en entrée)
    identification             système de Wiener-Hammerstein (filtre, non-linéarité statique, filtre)
    télécommunications         égalisation d'un canal non linéaire (Jaeger et Haas, 2004) ; prédiction d'un canal
                               sans fil à évanouissements de Rayleigh (CSI, modèle de Clarke) ; modèle d'amplificateur
                               de puissance (Saleh) avec mémoire
    bruit                      Mackey-Glass avec une entrée bruitée à 20 % (cible propre)
    peu de données             Mackey-Glass et NARMA10 avec 300 pas d'entraînement (au lieu de 3 000)
    mémoire longue             rappeler l'entrée d'il y a 30 ou 60 pas
"""
import numpy as np

from . import tasks

T_TEST = 1000


def _norm(u, y, nt):
    u = np.asarray(u, float)
    y = np.asarray(y, float)
    u = (u - u[:nt].mean(0)) / (u[:nt].std(0) + 1e-12)
    y = (y - y[:nt].mean()) / (y[:nt].std() + 1e-12)
    return u, y, nt


def _rk4(f, x, dt):
    k1 = f(x); k2 = f(x + dt / 2 * k1); k3 = f(x + dt / 2 * k2); k4 = f(x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def lorenz(T=4000, horizon=10, dt=0.01, sub=2, seed=0):
    """Composante x du système de Lorenz (σ = 10, ρ = 28, β = 8/3), échantillonnée tous les 0.02 ; prédire x à
    +10 échantillons (0.2 unité de temps, environ 0.2 temps de Lyapunov)."""
    f = lambda s: np.array([10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - 8 / 3 * s[2]])  # noqa: E731
    s = np.array([1.0, 1.0, 1.0]) + np.random.default_rng(seed).normal(0, 0.1, 3)
    out = []
    for i in range((T + horizon + 500) * sub):
        s = _rk4(f, s, dt)
        if i % sub == 0:
            out.append(s[0])
    x = np.array(out[500:])
    return _norm(x[:T], x[horizon:T + horizon], T - T_TEST)


def henon(T=4000, horizon=3, seed=0):
    """Application de Hénon (a = 1.4, b = 0.3) ; prédire x trois itérations plus loin."""
    x, y = 0.1 + 0.01 * seed, 0.0
    out = []
    for _ in range(T + horizon + 100):
        x, y = 1 - 1.4 * x * x + y, 0.3 * x
        out.append(x)
    x = np.array(out[100:])
    return _norm(x[:T], x[horizon:T + horizon], T - T_TEST)


def lorenz96(T=4000, horizon=5, D=40, F=8.0, dt=0.01, sub=5, seed=0):
    """Lorenz-96 à 40 variables (F = 8), échantillonné tous les 0.05 : les 40 variables en entrée, prédire la
    première 5 échantillons plus loin. Une entrée de grande dimension, mais déterministe."""
    def f(x):
        return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F
    x = F + np.random.default_rng(seed).normal(0, 0.5, D)
    out = []
    for i in range((T + horizon + 200) * sub):
        x = _rk4(f, x, dt)
        if i % sub == 0:
            out.append(x.copy())
    X = np.array(out[200:])
    return _norm(X[:T], X[horizon:T + horizon, 0], T - T_TEST)


def _iir(b, a, x):
    """Filtre IIR y[n] = Σ b_k x[n−k] − Σ_{k≥1} a_k y[n−k] (a[0] = 1)."""
    y = np.zeros_like(x)
    for n in range(len(x)):
        acc = sum(b[k] * x[n - k] for k in range(len(b)) if n - k >= 0)
        acc -= sum(a[k] * y[n - k] for k in range(1, len(a)) if n - k >= 0)
        y[n] = acc
    return y


def wiener_hammerstein(T=4000, noise=0.01, seed=7):
    """Filtre passe-bas → non-linéarité statique asymétrique → filtre résonant, plus un bruit de mesure de 1 %.
    Identification en simulation : la cible ne dépend que des entrées passées, jamais des sorties mesurées."""
    r = np.random.default_rng(seed)
    u = r.uniform(-1, 1, T + 200)
    v = _iir([0.2, 0.1], [1, -0.7], u)                                 # filtre d'entrée
    w = np.tanh(1.5 * v) + 0.3 * v ** 2                                 # non-linéarité statique
    y = _iir([0.3, 0.0, -0.3], [1, -1.4, 0.81], w)                      # filtre de sortie (résonance)
    y = y + noise * y.std() * r.normal(size=len(y))
    return _norm(u[200:], y[200:], T - T_TEST)


def channel_equalization(T=4000, snr_db=28, seed=3):
    """Égalisation de canal (Jaeger et Haas, Science 2004) : symboles d ∈ {−3, −1, 1, 3}, interférence entre
    symboles linéaire, distorsion non linéaire u = q + 0.036 q² − 0.011 q³ et bruit ; retrouver d(t − 2)."""
    r = np.random.default_rng(seed)
    d = r.choice([-3.0, -1.0, 1.0, 3.0], T + 20)
    h = {-2: 0.08, -1: -0.12, 0: 1.0, 1: 0.18, 2: -0.1, 3: 0.091, 4: -0.05, 5: 0.04, 6: 0.03, 7: 0.01}
    q = np.zeros(T + 20)
    for n in range(10, T + 10):
        q[n] = sum(c * d[n - k] for k, c in h.items())
    q = q[10:T + 10]
    u = q + 0.036 * q ** 2 - 0.011 * q ** 3
    u = u + r.normal(0, np.sqrt(u.var() / 10 ** (snr_db / 10)), T)
    target = d[8:T + 8]                                                  # d(n − 2), avec d décalé de 10
    return _norm(u, target, T - T_TEST)


def rayleigh_csi(T=4000, doppler=0.02, horizon=5, paths=32, snr_db=25, seed=5):
    """Canal sans fil à évanouissements de Rayleigh (modèle de Clarke, somme de 32 trajets, Doppler normalisé
    f_D·T_s = 0.02). Entrée : l'estimation bruitée du canal (parties réelle et imaginaire) ; cible : la partie
    réelle du vrai canal 5 pas plus loin. Le canal est un processus gaussien : un prédicteur linéaire est presque
    optimal, et l'accordé ne devrait pas gagner ici (contrôle)."""
    r = np.random.default_rng(seed)
    th = r.uniform(0, 2 * np.pi, paths)
    ph = r.uniform(0, 2 * np.pi, paths)
    t = np.arange(T + horizon)[:, None]
    h = np.exp(1j * (2 * np.pi * doppler * t * np.cos(th) + ph)).sum(1) / np.sqrt(paths)
    sig = np.sqrt(1 / 10 ** (snr_db / 10) / 2)
    est = h + sig * (r.normal(size=len(h)) + 1j * r.normal(size=len(h)))
    u = np.stack([est.real, est.imag], 1)[:T]
    return _norm(u, h.real[horizon:T + horizon], T - T_TEST)


def power_amplifier(T=4000, seed=9):
    """Amplificateur de puissance à mémoire (Wiener-Hammerstein-Saleh) : filtre d'entrée, distorsions d'amplitude
    et de phase de Saleh (1981), filtre de sortie. Entrée : le signal en bande de base (réel, imaginaire), proche
    de la saturation ; cible : la partie réelle de la sortie."""
    r = np.random.default_rng(seed)
    n = T + 50
    x = (r.normal(size=n) + 1j * r.normal(size=n)) / np.sqrt(2)
    x = np.convolve(x, np.ones(4) / 2, mode="same")                      # signal à bande limitée
    x = 0.8 * x / np.abs(x).std()
    x1 = np.convolve(x, [1.0, 0.3 + 0.1j, 0.1], mode="full")[:n]
    a = np.abs(x1)
    amp = 2.1587 * a / (1 + 1.1517 * a ** 2)
    phase = 4.0033 * a ** 2 / (1 + 9.1040 * a ** 2)
    z = amp * np.exp(1j * (np.angle(x1) + phase))
    y = np.convolve(z, [1.0, -0.2, 0.05j], mode="full")[:n]
    u = np.stack([x.real, x.imag], 1)[50:]
    return _norm(u, y.real[50:], T - T_TEST)


def noisy_mackey_glass(noise, T=4000):
    """Mackey-Glass (+15), entrée observée avec un bruit gaussien (écart type `noise` × celui du signal), cible
    propre : prédire la vraie série à partir de mesures bruitées."""
    u, y = tasks.task_mackey_glass(T)
    r = np.random.default_rng(21)
    u = u + noise * u.std() * r.normal(size=len(u))
    return _norm(u, y, T - T_TEST)


def few_data(name, n_train):
    """Mackey-Glass ou NARMA10 avec seulement n_train pas d'entraînement (test : toujours 1 000 pas)."""
    T = n_train + T_TEST
    u, y = tasks.task_mackey_glass(T) if name == "mg" else tasks.task_narma10(T)
    return _norm(u, y, n_train)


def long_delay(delay):
    u, y = tasks.task_delay(4000, delay=delay)
    return _norm(u, y, 3000)


# nom → (générateur, famille)
TASKS = {
    "Lorenz (+10)": (lorenz, "chaos"),
    "Hénon (+3)": (henon, "chaos"),
    "Lorenz-96, 40 entrées (+5)": (lorenz96, "chaos, grande dimension"),
    "Wiener-Hammerstein": (wiener_hammerstein, "identification"),
    "Égalisation de canal": (channel_equalization, "télécoms"),
    "Canal sans fil, CSI (+5)": (rayleigh_csi, "télécoms (contrôle)"),
    "Amplificateur de puissance": (power_amplifier, "télécoms"),
    "Mackey-Glass, entrée bruitée 20 %": (lambda: noisy_mackey_glass(0.2), "bruit"),
    "Mackey-Glass, 300 pas": (lambda: few_data("mg", 300), "peu de données"),
    "NARMA10, 300 pas": (lambda: few_data("narma", 300), "peu de données"),
    "Mémoire (délai 30)": (lambda: long_delay(30), "mémoire longue"),
    "Mémoire (délai 60)": (lambda: long_delay(60), "mémoire longue"),
}


def load(name):
    return TASKS[name][0]()


def linear_baseline(u, y, nt, lags=(5, 10, 20, 40, 80), lams=(1e-6, 1e-4, 1e-2, 1, 1e2), washout=100):
    """Régression ridge sur une fenêtre de valeurs passées de l'entrée (toutes les dimensions). La longueur de la
    fenêtre et la régularisation sont choisies sur une validation (les derniers 20 % de l'entraînement), puis le
    modèle est réappris sur tout l'entraînement. Renvoie (NMSE test, {"lags": .., "lam": ..})."""
    U = u if u.ndim == 2 else u[:, None]
    T, d = U.shape

    def feats(K):
        F = np.zeros((T, K * d + 1))
        for k in range(K):
            F[k:, k * d:(k + 1) * d] = U[:T - k]
        F[:, -1] = 1
        return F

    def fit(F, a, b, lam):
        A = F[a:b]
        return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y[a:b])

    nv = int(0.8 * nt)
    best = None
    for K in lags:
        if K * d + 1 > (nv - washout) // 2:
            continue
        F = feats(K)
        for lam in lams:
            w = fit(F, washout, nv, lam)
            e = np.mean((F[nv:nt] @ w - y[nv:nt]) ** 2)
            if best is None or e < best[0]:
                best = (e, K, lam)
    _, K, lam = best
    F = feats(K)
    w = fit(F, washout, nt, lam)
    p = F[nt:] @ w
    return float(np.mean((p - y[nt:]) ** 2) / y[nt:].var()), {"lags": K, "lam": lam}


def memory_polynomial_baseline(u, y, nt, memories=(2, 4, 6, 10), orders=(3, 5, 7, 9), cross=(0, 2),
                               lams=(1e-6, 1e-4, 1e-2, 1), washout=100):
    """Référence du domaine pour les amplificateurs de puissance : polynôme à mémoire généralisé (GMP, Kim et
    Konstantinou 2001 ; Morgan et al. 2006), sans état récurrent, résolu par ridge.

    L'entrée u (T, 2) est le signal complexe x = u[:, 0] + i·u[:, 1]. Caractéristiques :
        x(n − m) · |x(n − m − l)|^(k − 1),   m < M (mémoire), k impair ≤ K (ordre), |l| ≤ L (termes croisés),
    parties réelle et imaginaire séparées (la cible réelle est une combinaison linéaire réelle de celles-ci).
    L = 0 donne le polynôme à mémoire simple. M, K, L et la régularisation sont choisis sur une validation (les
    derniers 20 % de l'entraînement), puis le modèle est réappris sur tout l'entraînement.
    Renvoie (NMSE test, {"M": .., "K": .., "L": .., "lam": .., "caractéristiques": ..})."""
    x = u[:, 0] + 1j * u[:, 1]
    T = len(x)
    a = np.abs(x)

    def shift(v, s):                         # v(n − s), zéros hors de la série
        out = np.zeros_like(v)
        if s >= 0:
            out[s:] = v[:T - s] if s else v
        else:
            out[:s] = v[-s:]
        return out

    def feats(M, K, L):
        cols = []
        for m in range(M):
            xm = shift(x, m)
            for k in range(1, K + 1, 2):
                for l in (range(-L, L + 1) if k > 1 else (0,)):
                    c = xm * shift(a, m + l) ** (k - 1)
                    cols += [c.real, c.imag]
        return np.column_stack(cols + [np.ones(T)])

    def fit(F, a0, b0, lam):
        A = F[a0:b0]
        s = A.std(0) + 1e-12                 # mise à l'échelle des colonnes (ordres élevés), biais compris
        s[-1] = 1.0
        w = np.linalg.solve((A / s).T @ (A / s) + lam * np.eye(A.shape[1]), (A / s).T @ y[a0:b0])
        return w / s

    nv = int(0.8 * nt)
    best = None
    for M in memories:
        for K in orders:
            for L in cross:
                F = feats(M, K, L)
                if F.shape[1] > (nv - washout) // 2:
                    continue
                for lam in lams:
                    w = fit(F, washout, nv, lam)
                    e = np.mean((F[nv:nt] @ w - y[nv:nt]) ** 2)
                    if best is None or e < best[0]:
                        best = (e, M, K, L, lam, F.shape[1])
    _, M, K, L, lam, nf = best
    F = feats(M, K, L)
    w = fit(F, washout, nt, lam)
    p = F[nt:] @ w
    return float(np.mean((p - y[nt:]) ** 2) / y[nt:].var()), {"M": M, "K": K, "L": L, "lam": lam,
                                                              "caractéristiques": nf}
