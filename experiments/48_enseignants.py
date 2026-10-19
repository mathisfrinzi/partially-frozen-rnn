"""
Partie 9, étape 5 — un témoin positif : des tâches dont la structure est plantée.

Sur les tâches des parties 3 à 8, le gradient ne dit pas quelle structure apprendre (étapes 1 à 4). Mais rien ne dit que ces
tâches aient une structure à trouver. Ici, la structure est plantée dans un enseignant, et atteignable par construction.

Enseignants (100 neurones, même W_in que l'élève, biais nuls). Chaque enseignant s'écarte du réservoir de départ W₀ de l'élève
d'une distance fixée d'avance, ‖W_ens − W₀‖_F = a·‖W₀‖_F, a = 0,1 (régime du premier ordre) ou 0,5 :

    T1 tournée     e^{cΩ} W₀ e^{−cΩ}, Ω antisymétrique aléatoire         attendu : tourné
    T2 cisaillée   Q (R + cS), S strictement triangulaire aléatoire       attendu : cisaillé plein
    T3 accordée    U diag(σ ⊙ e^{c z}) Vᵀ, z aléatoire                    attendu : accordé
    T4 bande       Q (R + cS), S limitée aux 10 premières sur-diagonales   attendu : bande k = 10
    T5 témoin      W₀ + cP, P pleine aléatoire (aucune structure)          attendu : RNN

    (Q, R : QR de W₀, diagonale de R positive ; U, σ, V : SVD de W₀ ; mêmes conventions que les élèves.)
    Entrée u_t ~ N(0, 0,5²) i.i.d., 4 000 pas ; sortie y_t = lecture aléatoire fixée de l'état de l'enseignant, centrée réduite,
    plus un bruit de mesure gaussien à 30 dB (plancher de NMSE : 10⁻³). L'élève prédit y_t à partir de u_≤t. Découpage et
    entraînement de la partie 9 : 75 % d'entraînement (dont le dernier quart sert de validation aux signaux), 25 % de test,
    1 500 pas d'Adam.

Contrôles de construction, avant tout le reste
    oracle         réservoir figé W = W_ens, lecture ridge seule : doit atteindre le plancher du bruit (≈ −3 en log10)
    classique      réservoir figé W₀, lecture ridge seule : ce que l'élève doit rattraper
    bases          chaque base orthonormée B_s vérifie ‖B_sᵀB_s − I‖_max < 1e-5 (sur 500 colonnes tirées), la stabilité
                   ‖B_s B_sᵀ v − v‖ / ‖v‖ < 1e-5 pour v dans l'espace, et sa dimension est comparée au nombre de paramètres.
                   Une structure qui rate (a) ou (b) est écartée des signaux (signalé dans le rapport).
    vérité terrain la structure attendue doit gagner sa tâche ; sinon la tâche est signalée « mal calibrée » dans le rapport,
                   et l'enseignant n'est PAS retouché (l'amplitude est fixée d'avance).

Signaux au départ (même protocole que l'étape 1, 128 fenêtres d'entraînement et 128 de validation)
    complet        les six signaux de l'étape 1 sur (W, W_in, b)
    récurrent      les mêmes sur la partie récurrente seule (W), sans biais ni W_in
    PCA            directions propres de la covariance des tangents, sur 1 024 fenêtres : X = G (gradient brut) ou
                   X = M⁻¹G régularisé (ε = 0,1), non centrée ou centrée ; pour chaque structure, la part de la 1ʳᵉ direction
                   (ramenée à ΔW) dans T_s, divisée par la part attendue au hasard dim(T_s)/N²

Critères, fixés d'avance
    choix          parmi les six structures récurrentes autres que le RNN (accordé, tourné, cisaillé plein, bandes k = 3 et
                   10, nilpotent) ; le RNN et le réservoir à biais et entrées restent dans la vérité terrain.
    T1–T4          le signal choisit-il la structure attendue ? 8 cas (4 tâches × 2 amplitudes), hasard : 8/6 ≈ 1,3.
                   O si 6 cas sur 8 ou plus, partiel si 3 à 5, N sinon.
    T5             le signal choisit-il la même structure sur les trois graines ? Sur une tâche sans structure, un choix
                   constant vient de la taille ou de W₀, pas de la tâche.
    verdict        « le signal géométrique existe quand la géométrie existe » si au moins un signal obtient O ;
                   sinon « même sur des tâches géométriques, le gradient au départ ne dit rien ».

    python experiments/48_enseignants.py --seed 0 --threads 1      # une graine, écrit enseignants_s0.json
    python experiments/48_enseignants.py --report
"""
import argparse
import gc
import glob
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn import core

TASKS = ["T1 tournée", "T2 cisaillée", "T3 accordée", "T4 bande", "T5 témoin"]
AMPS = [0.1, 0.5]
NAMES = ["RNN", "réservoir accordé (valeurs singulières)", "réservoir tourné (rotations)", "Iwasawa : cisaillement (N)",
         "réservoir cisaillé en bande, k = 3", "réservoir cisaillé en bande, k = 10",
         "Jordan-Chevalley + partie nilpotente apprise", "réservoir à biais et entrées appris (sans g)"]
SHORT = {NAMES[0]: "RNN", NAMES[1]: "accordé", NAMES[2]: "tourné", NAMES[3]: "cisaillé plein", NAMES[4]: "bande k=3",
         NAMES[5]: "bande k=10", NAMES[6]: "nilpotent", NAMES[7]: "biais et entrées"}
EXPECTED = {"T1 tournée": NAMES[2], "T2 cisaillée": NAMES[3], "T3 accordée": NAMES[1], "T4 bande": NAMES[5],
            "T5 témoin": NAMES[0]}
CHOICE = NAMES[1:7]                                   # les six structures récurrentes autres que le RNN
GRAD = ["part captée", "part par paramètre", "signal sur bruit", "accord entraînement-validation", "pas de Newton",
        "directions principales"]
PCA_VARIANTS = [("G brut", False), ("G brut", True), ("M⁻¹G (ε = 0.1)", False), ("M⁻¹G (ε = 0.1)", True)]
T_LEN, U_STD, SNR_DB, N_PCA, B = 4000, 0.5, 30.0, 1024, 128


def key(task, amp):
    return f"{task} | a = {amp}"


def structures():
    from hfrnn.structure_signals import STRUCTURES
    d = dict(STRUCTURES)
    return [(s, d[s]) for s in NAMES]


# --------------------------------------------------------------------------------------------- enseignants
def qr_pos(W):
    Q, R = np.linalg.qr(W)
    s = np.where(np.diag(R) >= 0, 1.0, -1.0)
    return Q * s, R * s[:, None]


def _bisect(dist, target, c_hi=1.0):
    """c ≥ 0 tel que dist(c) = target (dist croissante près de 0)."""
    while dist(c_hi) < target and c_hi < 1e3:
        c_hi *= 2
    lo, hi = 0.0, c_hi
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if dist(mid) < target else (lo, mid)
    return (lo + hi) / 2


def teacher(task, amp, W0, seed):
    """Matrice de l'enseignant, à la distance amp·‖W₀‖_F de W₀, dans la structure plantée."""
    from scipy.linalg import expm
    n = W0.shape[0]
    rng = np.random.default_rng(4800 + 17 * seed + TASKS.index(task))
    target = amp * np.linalg.norm(W0)
    if task == "T1 tournée":
        A = rng.normal(size=(n, n))
        Om = (A - A.T) / np.sqrt(2 * n)
        rot = lambda c: expm(c * Om) @ W0 @ expm(-c * Om)                    # noqa: E731
        c = _bisect(lambda c: np.linalg.norm(rot(c) - W0), target)
        return rot(c)
    if task in ("T2 cisaillée", "T4 bande"):
        Q, R = qr_pos(W0)
        S = np.triu(rng.normal(size=(n, n)), 1)
        if task == "T4 bande":
            S = np.triu(S, 1) - np.triu(S, 11)                               # 10 premières sur-diagonales
        return Q @ (R + target / np.linalg.norm(S) * S)                       # ‖Q·cS‖ = c‖S‖
    if task == "T3 accordée":
        U, sig, Vt = np.linalg.svd(W0)
        z = rng.normal(size=n)
        c = _bisect(lambda c: np.linalg.norm(sig * (np.exp(c * z) - 1)), target)
        return (U * (sig * np.exp(c * z))) @ Vt
    if task == "T5 témoin":
        P = rng.normal(size=(n, n))
        return W0 + target / np.linalg.norm(P) * P
    raise ValueError(task)


def make_data(task, amp, Win, W0, seed):
    """(u, y, nt, W_ens) : l'enseignant piloté par u i.i.d., sortie lue au hasard, centrée réduite, bruit à 30 dB."""
    Wt = teacher(task, amp, W0, seed)
    rng = np.random.default_rng(4900 + 17 * seed + TASKS.index(task) + int(100 * amp))
    u = rng.normal(0, U_STD, T_LEN)
    h, H = np.zeros(W0.shape[0]), np.empty((T_LEN, W0.shape[0]))
    for t in range(T_LEN):
        h = np.tanh(Win[:, 0] * u[t] + Wt @ h)
        H[t] = h
    y = H @ rng.normal(size=W0.shape[0])
    y = (y - y.mean()) / y.std()
    y = y + rng.normal(0, 10 ** (-SNR_DB / 20), T_LEN)
    return u, y, core.n_train(u), Wt


# --------------------------------------------------------------------------------------------- contrôles des bases
def check_basis(U, n_params, seed):
    """(a) orthonormalité sur 500 colonnes, (b) stabilité de la projection, (c) dimension ; renvoie un dict."""
    if isinstance(U, str) or U is None:
        return {"ok": True, "dimension": None}
    rng = np.random.default_rng(seed)
    cols = rng.choice(U.shape[1], min(500, U.shape[1]), replace=False)
    Ud = U[:, cols].astype(np.float64)
    ortho = float(np.abs(Ud.T @ Ud - np.eye(len(cols))).max())
    v = U.astype(np.float64) @ rng.normal(size=U.shape[1])
    stab = float(np.linalg.norm(U.astype(np.float64) @ (U.T.astype(np.float64) @ v) - v) / np.linalg.norm(v))
    return {"ok": ortho < 1e-5 and stab < 1e-5, "orthonormalité": ortho, "stabilité": stab,
            "dimension": int(U.shape[1]), "paramètres": int(n_params)}


# --------------------------------------------------------------------------------------------- signaux
def gradient_signals(u, y, nt, Win, W0, seed, bases):
    """{"complet": {s: signaux}, "récurrent": {s: signaux}} au réservoir de départ."""
    from hfrnn import structure_signals as SS
    w, c = SS.ridge_readout(u, y, nt, Win, W0)
    nv = int(0.75 * nt)
    rng = np.random.default_rng(500 + seed)
    tr = rng.integers(0, nv - core.WIN_LEN - 1, B)
    va = rng.integers(nv, nt - core.WIN_LEN - 1, B)
    gtr, gva = SS.window_grads(u, y, tr, Win, W0, w, c), SS.window_grads(u, y, va, Win, W0, w, c)
    out = {}
    for variant in ("complet", "récurrent"):
        if variant == "complet":
            Gtr = np.concatenate(gtr, 1)
            parts = lambda g, P: SS.coords(*g, P)                                          # noqa: E731
        else:
            Gtr = gtr[0]
            parts = lambda g, P: g[0] if isinstance(P, str) else g[0] @ P                  # noqa: E731
        lam = float((Gtr ** 2).sum(1).mean() / Gtr.shape[1])
        _, _, Vt = np.linalg.svd(Gtr, full_matrices=False)
        pcs = Vt[:10]
        nW, nI = W0.shape[0] ** 2, Win.size
        res = {}
        for s, P in bases.items():
            if variant == "récurrent" and P is None:
                continue                                                  # aucune direction récurrente
            Ztr, Zva = parts(gtr, P), parts(gva, P)
            if variant == "complet":
                Pp = SS.coords(pcs[:, :nW], pcs[:, nW:nW + nI], pcs[:, nW + nI:], P)
            else:
                Pp = pcs if isinstance(P, str) else pcs @ P
            res[s] = SS.signals(Ztr, Zva, Gtr, lam, Pp)
        out[variant] = res
    return out


def pca_alignment(u, y, nt, Win, W0, seed, bases):
    """Spectre de la covariance des tangents et alignement de ses premières directions avec chaque T_s."""
    from hfrnn import structure_signals as SS
    import importlib.util
    spec = importlib.util.spec_from_file_location("m47", Path(__file__).with_name("47_protocole_lie.py"))
    m47 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m47)
    w, c = SS.ridge_readout(u, y, nt, Win, W0)
    rng = np.random.default_rng(600 + seed)
    starts = rng.integers(0, int(0.75 * nt) - core.WIN_LEN - 1, N_PCA)
    G = SS.window_grads(u, y, starts, Win, W0, w, c)[0]
    n2 = W0.shape[0] ** 2
    out = {}
    for tg, centred in PCA_VARIANTS:
        X = m47.tangent_samples(G, W0, tg)
        if centred:
            X = X - X.mean(0, keepdims=True)
        _, s, Vt = np.linalg.svd(X.astype(np.float32), full_matrices=False)
        ev = s ** 2 / len(X)
        D = m47.to_weights(Vt[:5], W0, tg)                               # directions ramenées à ΔW
        D = D / np.linalg.norm(D, axis=1, keepdims=True)
        al = {}
        for sname, P in bases.items():
            if isinstance(P, str):
                part, dim = np.ones(5), n2
            elif P is None:
                part, dim = np.zeros(5), 0
            else:
                part, dim = ((D @ P) ** 2).sum(1), P.shape[1]
            al[sname] = {"part 1re": float(part[0]), "part 5 premières": float(part.mean()),
                         "hasard": dim / n2, "rapport 1re": float(part[0] / (dim / n2)) if dim else 0.0}
        out[f"{tg}{', centrée' if centred else ''}"] = {"spectre": [float(x) for x in ev[:20] / ev.sum()], "alignement": al}
    return out


# --------------------------------------------------------------------------------------------- calcul
def run(a, path):
    import torch
    from hfrnn import structure_signals as SS
    from hfrnn import torch_models as TM
    torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    res = json.load(open(path)) if path.exists() else {"args": {"n": a.n, "steps": a.steps}, "bases": {}, "tâches": {}}
    t0 = time.time()
    Win, W0 = SS.reservoir(a.n, 1, a.seed)
    S = structures()
    bases = None
    for amp in a.amps:
        for task in (a.tasks or TASKS):
            k = key(task, amp)
            cell = res["tâches"].setdefault(k, {})
            u, y, nt, Wt = make_data(task, amp, Win, W0, a.seed)
            if "contrôles" not in cell:
                v_or, _ = TM.train_regression(u, y, Win, Wt, a.seed, kind="reservoir", n_train=nt)
                v_cl, _ = TM.train_regression(u, y, Win, W0, a.seed, kind="reservoir", n_train=nt)
                cell["contrôles"] = {"oracle": float(np.log10(v_or)), "classique": float(np.log10(v_cl)),
                                     "distance relative": float(np.linalg.norm(Wt - W0) / np.linalg.norm(W0)),
                                     "rayon spectral enseignant": float(max(abs(np.linalg.eigvals(Wt))))}
            gt = cell.setdefault("vérité terrain", {})
            for s, kw in S:                                                # structures en boucle interne
                if s in gt:
                    continue
                try:
                    v, info = TM.train_regression(u, y, Win, W0, a.seed, steps=a.steps, device=device, n_train=nt,
                                                  **dict(kw))
                    gt[s] = float(np.log10(v)) if np.isfinite(v) else float("inf")
                except (RuntimeError, ValueError):
                    gt[s] = float("inf")
                json.dump(res, open(path, "w"), indent=1, ensure_ascii=False)
            if "signaux" not in cell or "pca" not in cell:
                if bases is None:
                    bases = {}
                    for s, kw in S:
                        J = SS.tangent_jacobian(Win, W0, a.seed, kw)
                        bases[s] = SS.orthonormal_basis(J)
                        n_par = 0 if J is None or isinstance(J, str) else J.shape[1]
                        res["bases"].setdefault(s, check_basis(bases[s], n_par, a.seed))
                        del J
                        gc.collect()
                    bad = [s for s, v in res["bases"].items() if not v["ok"]]
                    if bad:
                        print("BASES REFUSÉES (écartées des signaux) :", ", ".join(bad), flush=True)
                    json.dump(res, open(path, "w"), indent=1, ensure_ascii=False)
                ok = {s: P for s, P in bases.items() if res["bases"][s]["ok"]}
                cell["signaux"] = gradient_signals(u, y, nt, Win, W0, a.seed, ok)
                cell["pca"] = pca_alignment(u, y, nt, Win, W0, a.seed, ok)
                json.dump(res, open(path, "w"), indent=1, ensure_ascii=False)
            c = cell["contrôles"]
            print(f"[{(time.time() - t0) / 60:5.1f} min] {k:22s} graine {a.seed} : oracle {c['oracle']:+.2f}, "
                  f"classique {c['classique']:+.2f}, meilleure {SHORT[min(gt, key=gt.get)]}", flush=True)


# --------------------------------------------------------------------------------------------- analyse
def load(pattern):
    out = {}
    for f in sorted(glob.glob(pattern)):
        seed = Path(f).stem.rsplit("_s", 1)[1]
        r = json.load(open(f))
        for k, cell in r["tâches"].items():
            if len(cell.get("vérité terrain", {})) == len(NAMES) and "pca" in cell:
                out.setdefault(k, {})[seed] = cell
        out.setdefault("_bases", {})[seed] = r.get("bases", {})
    return out


def signal_values(cell, sig):
    """{structure: valeur} pour un signal ; None si le signal n'est pas défini pour la structure."""
    if sig == "taille":
        return {s: v["dimension"] for s, v in cell["signaux"]["complet"].items()}
    if sig.startswith("PCA "):
        var = sig[4:].split(" : ")[0]
        return {s: v["rapport 1re"] for s, v in cell["pca"][var]["alignement"].items()}
    variant, name = sig.split(" : ")
    return {s: v[name] for s, v in cell["signaux"][variant].items()}


def all_signals():
    return ([f"complet : {g}" for g in GRAD] + [f"récurrent : {g}" for g in GRAD]
            + [f"PCA {tg}{', centrée' if c else ''} : alignement / hasard" for tg, c in PCA_VARIANTS] + ["taille"])


def evaluate(R):
    from scipy.stats import spearmanr
    cells = [k for k in R if not k.startswith("_")]
    order = [key(t, a) for a in AMPS for t in TASKS if key(t, a) in cells]
    out = {"cellules": order, "vérité terrain": {}, "calibrage": {}, "signaux": {}}
    for k in order:
        seeds = sorted(R[k])
        gt = {s: float(np.mean([R[k][sd]["vérité terrain"][s] for sd in seeds])) for s in NAMES}
        ctrl = {c: float(np.mean([R[k][sd]["contrôles"][c] for sd in seeds])) for c in R[k][seeds[0]]["contrôles"]}
        task = k.split(" | ")[0]
        exp = EXPECTED[task]
        best_all = min(gt, key=gt.get)
        best_choice = min(CHOICE, key=gt.get)
        out["vérité terrain"][k] = {"gain": gt, "contrôles": ctrl, "attendue": exp, "meilleure": best_all,
                                    "meilleure parmi les six": best_choice}
        out["calibrage"][k] = (best_all == exp) if task == "T5 témoin" else (best_choice == exp)
    for sig in all_signals() + ["choix naïf"]:
        row = {}
        for k in order:
            seeds = sorted(R[k])
            gt = out["vérité terrain"][k]["gain"]
            if sig == "choix naïf":                                     # rang moyen sur les autres cellules
                others = [o for o in order if o != k]
                score = {s: -np.mean([out["vérité terrain"][o]["gain"][s] for o in others]) for s in CHOICE}
                per_seed = [score] * len(seeds)
            else:
                per_seed = [signal_values(R[k][sd], sig) for sd in seeds]
            names = [s for s in CHOICE if all(s in p for p in per_seed)]
            if not names:
                continue
            V = np.array([[p[s] for p in per_seed] for s in names], float)
            ranks = np.argsort(np.argsort(V, 0), 0)
            picks = [names[int(np.argmax(V[:, j]))] for j in range(V.shape[1])]
            chosen = names[int(np.argmax(ranks.mean(1)))]
            best = min(names, key=gt.get)
            rho = [spearmanr(V[:, j], [-gt[s] for s in names])[0] for j in range(V.shape[1])] if sig != "choix naïf" else []
            row[k] = {"choisie": chosen, "regret": gt[chosen] - gt[best], "attendue": chosen == EXPECTED[k.split(" | ")[0]],
                      "graines d'accord": len(set(picks)) == 1, "rho": float(np.nanmean(rho)) if rho else None}
        planted = [k for k in row if not k.startswith("T5")]
        hits = sum(row[k]["attendue"] for k in planted)
        verdict = "O" if hits >= 6 else ("partiel" if hits >= 3 else "N")
        t5 = [k for k in row if k.startswith("T5")]
        out["signaux"][sig] = {"cellules": row, "attendues trouvées": hits, "sur": len(planted), "verdict": verdict,
                               "T5 choix constant": sum(row[k]["graines d'accord"] for k in t5), "T5 cellules": len(t5),
                               "regret moyen": float(np.mean([v["regret"] for v in row.values()]))}
    # information propre à la tâche : signal × gain doublement centrés, p par permutation (structures, dans chaque cellule)
    rng = np.random.default_rng(0)
    dc = lambda M: M - M.mean(0, keepdims=True) - M.mean(1, keepdims=True) + M.mean()   # noqa: E731
    for sig in all_signals():
        if sig == "taille" or sig not in out["signaux"]:
            continue
        ks = list(out["signaux"][sig]["cellules"])
        try:
            X = np.array([[np.mean([signal_values(R[k][sd], sig)[s] for sd in R[k]]) for s in CHOICE] for k in ks])
        except KeyError:
            continue
        if "accord" not in sig:
            X = np.log(np.abs(X) + 1e-300)
        P = np.array([[-out["vérité terrain"][k]["gain"][s] for s in CHOICE] for k in ks])
        r = float(np.corrcoef(dc(X).ravel(), dc(P).ravel())[0, 1])
        null = [np.corrcoef(dc(np.array([row_[rng.permutation(len(CHOICE))] for row_ in X])).ravel(), dc(P).ravel())[0, 1]
                for _ in range(2000)]
        out["signaux"][sig]["interaction"] = {"r": r, "p": float(np.mean(np.abs(null) >= abs(r)))}
    good = [s for s, v in out["signaux"].items() if v["verdict"] == "O" and s not in ("taille", "choix naïf")]
    out["verdict"] = ("validé : le signal géométrique existe quand la géométrie existe (" + ", ".join(good) + ")") if good \
        else "invalidé : même sur des tâches géométriques, le gradient au départ ne dit pas quelle structure apprendre"
    return out


def report(a):
    R = load(str(core.out_path("data", "enseignants_s[0-9]*.json")))
    out = evaluate(R)
    order = out["cellules"]
    print(f"\n{len(order)} cellules (tâche × amplitude), graines : "
          + ", ".join(f"{k} {len(R[k])}" for k in order))
    bases = R.get("_bases", {})
    for sd, B_ in bases.items():
        bad = [SHORT[s] for s, v in B_.items() if not v["ok"]]
        print(f"bases, graine {sd} : " + ("toutes valides" if not bad else "REFUSÉES : " + ", ".join(bad))
              + " ; dimensions " + ", ".join(f"{SHORT[s]} {v['dimension']}/{v.get('paramètres', '—')}" for s, v in B_.items()
                                             if v.get("dimension")))
    print("\nContrôles et vérité terrain (log10 NMSE de test, moyenne sur les graines ; plancher du bruit ≈ −3) :")
    print(f"  {'':22s} {'oracle':>7s} {'classique':>9s} " + " ".join(f"{SHORT[s]:>10s}" for s in NAMES) + "   attendue gagne ?")
    for k in order:
        v = out["vérité terrain"][k]
        g = v["gain"]
        print(f"  {k:22s} {v['contrôles']['oracle']:>7.2f} {v['contrôles']['classique']:>9.2f} "
              + " ".join(f"{g[s]:>10.2f}" for s in NAMES)
              + f"   {'oui' if out['calibrage'][k] else 'NON (tâche mal calibrée)'} ; meilleure : {SHORT[v['meilleure']]}")
    print("\nSignaux : structure choisie parmi les six (accordé, tourné, cisaillé plein, bandes 3 et 10, nilpotent)")
    print(f"  {'signal':52s} {'attendue trouvée':>16s} {'verdict':>8s} {'T5 constant':>11s} {'regret':>7s} {'interaction':>18s}")
    for sig, v in out["signaux"].items():
        inter = v.get("interaction")
        print(f"  {sig:52s} {v['attendues trouvées']:>8d} / {v['sur']:<5d} {v['verdict']:>8s} "
              f"{v['T5 choix constant']:>5d} / {v['T5 cellules']:<3d} {v['regret moyen']:>7.2f} "
              + (f"r = {inter['r']:+.2f}, p = {inter['p']:.3f}" if inter else ""))
    print("\nChoix par cellule :")
    for sig, v in out["signaux"].items():
        print(f"  {sig} : " + " ; ".join(f"{k} → {SHORT[c['choisie']]}" for k, c in v["cellules"].items()))
    print("\nPCA : part de la trace dans la 1re direction et alignement de la 1re direction avec la structure attendue "
          "(rapport au hasard), moyenne sur les graines :")
    for k in order:
        exp = EXPECTED[k.split(" | ")[0]]
        parts = []
        for tg, c in PCA_VARIANTS:
            var = f"{tg}{', centrée' if c else ''}"
            sp = np.mean([R[k][sd]["pca"][var]["spectre"][0] for sd in R[k]])
            al = np.mean([R[k][sd]["pca"][var]["alignement"].get(exp, {}).get("rapport 1re", np.nan) for sd in R[k]])
            parts.append(f"{var} : {sp:.2f} / ×{al:.1f}")
        print(f"  {k:22s} " + " ; ".join(parts))
    print(f"\nVERDICT : {out['verdict']}")
    json.dump(out, open(core.out_path("data", "enseignants_summary.json"), "w"), indent=1, ensure_ascii=False, default=str)
    figures(R, out)


def figures(R, out):
    order = out["cellules"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 0.45 * len(order) + 2.2), gridspec_kw={"width_ratios": [1.3, 1]})
    M = np.array([[out["vérité terrain"][k]["gain"][s] - min(out["vérité terrain"][k]["gain"].values()) for s in NAMES]
                  for k in order])
    im = ax[0].imshow(np.minimum(M, 1.5), cmap="viridis_r", aspect="auto")
    ax[0].set_xticks(range(len(NAMES))); ax[0].set_xticklabels([SHORT[s] for s in NAMES], rotation=40, ha="right", fontsize=8)
    ax[0].set_yticks(range(len(order))); ax[0].set_yticklabels(order, fontsize=8)
    for i, k in enumerate(order):
        j = NAMES.index(EXPECTED[k.split(" | ")[0]])
        ax[0].add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="red", lw=1.5))
    fig.colorbar(im, ax=ax[0], label="écart à la meilleure (log10 NMSE)")
    ax[0].set_title("Vérité terrain (cadre rouge : structure plantée)", fontsize=10)
    sigs = [s for s in out["signaux"]]
    ax[1].barh(range(len(sigs)), [out["signaux"][s]["attendues trouvées"] for s in sigs], color="tab:blue")
    ax[1].axvline(8 / 6, color="gray", ls="--", label="hasard")
    ax[1].axvline(6, color="tab:green", ls=":", label="seuil « O »")
    ax[1].set_yticks(range(len(sigs))); ax[1].set_yticklabels(sigs, fontsize=7); ax[1].invert_yaxis()
    ax[1].set_xlabel("structure plantée retrouvée (cas sur 8)"); ax[1].legend(fontsize=7)
    ax[1].set_title("Le signal retrouve-t-il la structure plantée ?", fontsize=10)
    fig.tight_layout(); fig.savefig(core.out_path("figures", "enseignants.png"), dpi=110)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--steps", type=int, default=core.STEPS)
    p.add_argument("--amps", type=float, nargs="*", default=AMPS)
    p.add_argument("--tasks", nargs="*")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    if not a.report:
        run(a, core.out_path("data", f"enseignants_s{a.seed}.json"))
    else:
        report(a)
