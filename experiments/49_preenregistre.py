"""
Partie 9.7 — les deux indices de la partie 9 vérifiés sur des graines neuves, avec des critères fixés avant le calcul.

Phase 1, calibrage (graine 3, tâches plantées T1–T4 de l'étape 5, amplitudes a ∈ {0,2 ; 0,3 ; 0,4})
    Règle : a* est le plus petit a tel que, sur chacune des tâches T1 à T4,
        (i)  le réservoir figé W₀ (lecture seule) est à plus de 0,5 du plancher (log10 NMSE classique − oracle > 0,5) ;
        (ii) la structure plantée bat chacune des sept autres structures d'au moins 0,1 en log10 NMSE de test.
    La seconde amplitude est la valeur suivante de la grille {0,2 ; 0,3 ; 0,4 ; 0,5}.
    Issue de secours (déclarée d'avance) : si aucun a ne satisfait la règle, a* est celui qui maximise l'avance moyenne de la
    structure plantée sur la meilleure des autres (moyenne sur T1–T4), et le rapport le signale comme « recalibrage ». Le
    rapport dit aussi quelle condition a raté, et combien de fois : (ii) qui rate toujours veut dire que la batterie de
    structures ne départage pas une géométrie réelle ; (i) qui rate toujours, que les tâches sont trop faciles.

Phase 2, P2 (graines 3, 4, 5 ; les cinq tâches réelles ; protocole de l'étape 4 : 2 560 fenêtres, gradient brut, covariance
non centrée, 1 500 pas sans arrêt anticipé, le protocole où l'indice a été trouvé)
    P2 : avec d = 1 000, en moyenne sur les trois graines,
        les premières directions propres battent la bande k = 10 sur Lorenz et sur Hénon (2 conditions),
        et font moins bien que d = 1 000 directions aléatoires sur Mackey-Glass, NARMA10 et la mémoire (3 conditions).
    P2 passe si les 5 conditions sont remplies.
    Analyse post-hoc, assumée comme telle (descriptive, aucun test) : pour chaque tâche, (a) la part du déplacement d'un RNN
    entraîné, ΔW, qui tombe dans le sous-espace des 1 000 premières directions (comparée au hasard, 0,1) ; (b) le rang stable de
    ΔW, ‖ΔW‖²_F / ‖ΔW‖²₂ ; (c) l'écart à la normalité (Henrici) de W₀ + ΔW et de W₀ ; (d) le ratio de participation des valeurs
    propres de la covariance des tangents.

Phase 3, P1 (graines 4, 5, 6 ; tâches plantées T1–T4 et le témoin T5 aux deux amplitudes de la phase 1)
    P1 : le signal sur bruit restreint à la récurrence, mesuré au départ (protocole de l'étape 5), choisit la structure plantée
    (parmi les six structures récurrentes autres que le RNN, rang moyen sur les trois graines) dans au moins 5 des 8 cas
    (T1–T4 × 2 amplitudes). Le hasard en trouve 8/6 ≈ 1,3 ; 5 ou plus par hasard : p ≈ 0,005.
    Les autres signaux de l'étape 5 sont calculés et rapportés, sans être des prédictions.
    La vérité terrain des graines 4 à 6 est calculée après les signaux, pour vérifier que le calibrage tient sur les graines de
    test (ce n'est pas une condition de P1).

Entraînement des phases 1 et 3 : même budget pour toutes les structures, 3 000 pas d'Adam (lr de la partie 9), fenêtres tirées
dans les 75 % premiers de la partie entraînement, erreur de validation (lecture ridge, 25 % suivants) mesurée tous les 100 pas,
et retour aux poids du meilleur point de validation (arrêt anticipé) ; lecture ridge refaite sur toute la partie
entraînement, NMSE de test sur la partie test.

    python experiments/49_preenregistre.py --phase calibrage --amps 0.2      # puis 0.3, 0.4 (en parallèle possible)
    python experiments/49_preenregistre.py --phase p2 --seed 3               # puis 4, 5
    python experiments/49_preenregistre.py --phase p1 --seed 4               # signaux ; puis 5, 6, l'une après l'autre (mémoire)
    python experiments/49_preenregistre.py --phase p1 --seed 4 --part vérité # vérité terrain des graines de test
    python experiments/49_preenregistre.py --phase rapport
"""
import argparse
import copy
import gc
import glob
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hfrnn import core

GRID = [0.2, 0.3, 0.4]
NEXT = {0.2: 0.3, 0.3: 0.4, 0.4: 0.5}
PLANTED = ["T1 tournée", "T2 cisaillée", "T3 accordée", "T4 bande"]
CAL_SEED, P2_SEEDS, P1_SEEDS = 3, [3, 4, 5], [4, 5, 6]
STEPS_ES, EVERY = 3000, 100
REAL = ["Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)", "Lorenz (+10)", "Hénon (+3)"]
BAND_WINS = ["Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)"]
BAND_LOSES = ["Lorenz (+10)", "Hénon (+3)"]
P1_SIGNAL = "récurrent : signal sur bruit"


def _load(name, tag):
    spec = importlib.util.spec_from_file_location(tag, Path(__file__).with_name(name))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def m48():
    return _load("48_enseignants.py", "m48")


def save(res, path):
    json.dump(res, open(path, "w"), indent=1, ensure_ascii=False)


# --------------------------------------------------------------------------------------------- entraînement à arrêt anticipé
def fit_es(u, y, nt, Win, W, seed, kw, steps=STEPS_ES, every=EVERY, device="cpu"):
    """Même budget pour toutes les structures, retour au meilleur point de validation. Renvoie un dict (log10 NMSE de test,
    log10 NMSE de validation au meilleur point, pas retenu)."""
    import torch
    from hfrnn import torch_models as TM
    kw = dict(kw)
    kind = kw.pop("kind")
    dtype = torch.float32
    rng = np.random.default_rng(1000 + seed)
    model = TM.build_model(Win, W, kind, seed, dtype, device, **kw)
    n = model.n_features if hasattr(model, "n_features") else W.shape[0]
    ut = torch.as_tensor(u, dtype=dtype, device=device)
    yt = torch.as_tensor(y, dtype=dtype, device=device)
    nv, WO = int(0.75 * nt), core.WASHOUT

    @torch.no_grad()
    def score(fit_end, lo, hi):
        X = model(TM.as_batch(ut[:hi]))[0]
        w = TM.ridge(X[WO:fit_end], yt[WO:fit_end])
        Xd, yd = X.double(), yt[:hi].double()
        pred = torch.cat([Xd, torch.ones(len(Xd), 1, dtype=Xd.dtype, device=Xd.device)], 1) @ w
        return float(((pred[lo:hi] - yd[lo:hi]) ** 2).mean() / yd[lo:hi].var(unbiased=False))

    best = (score(nv, nv, nt), 0, copy.deepcopy(model.state_dict()))
    if kind != "reservoir" and steps > 0:
        Wout = torch.nn.Parameter(torch.zeros(n, 1, dtype=dtype, device=device))
        c = torch.nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
        others = ([model.bias] if model.bias.requires_grad else []) + [Wout, c]
        opt = torch.optim.Adam(model.param_groups(core.LR) + [{"params": others, "lr": core.LR}], betas=(0.9, 0.999), eps=1e-8)
        clip = model.recurrent_params() + others
        for step in range(steps):
            starts = rng.integers(0, nv - core.WIN_LEN - 1, core.BATCH)
            idx = torch.as_tensor(starts[:, None] + np.arange(core.WIN_LEN)[None, :], device=device)
            U = ut[idx].unsqueeze(-1)
            pred = (model(U) @ Wout)[..., 0] + c
            loss = ((pred[:, core.LOSS_WASH:] - yt[idx][:, core.LOSS_WASH:]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            gn = torch.sqrt(sum((p.grad ** 2).sum() for p in clip if p.grad is not None))
            if gn > 1.0:
                for p in clip:
                    if p.grad is not None:
                        p.grad /= gn
            opt.step()
            if (step + 1) % every == 0:
                v = score(nv, nv, nt)
                if np.isfinite(v) and v < best[0]:
                    best = (v, step + 1, copy.deepcopy(model.state_dict()))
        model.load_state_dict(best[2])
    test = score(nt, nt, len(u))
    lg = lambda x: float(np.log10(x)) if np.isfinite(x) and x > 0 else float("inf")   # noqa: E731
    return {"test": lg(test), "validation": lg(best[0]), "pas retenu": int(best[1])}


def ground_truth(task, amp, seed, device, cell, path, res):
    """Oracle, classique et huit structures (arrêt anticipé) pour une tâche plantée ; sauvegarde après chaque entraînement."""
    from hfrnn import structure_signals as SS
    M = m48()
    Win, W0 = SS.reservoir(100, 1, seed)
    u, y, nt, Wt = M.make_data(task, amp, Win, W0, seed)
    if "oracle" not in cell:
        cell["oracle"] = fit_es(u, y, nt, Win, Wt, seed, {"kind": "reservoir"}, device=device)
        cell["classique"] = fit_es(u, y, nt, Win, W0, seed, {"kind": "reservoir"}, device=device)
        save(res, path)
    gt = cell.setdefault("structures", {})
    for s, kw in M.structures():
        if s in gt:
            continue
        try:
            gt[s] = fit_es(u, y, nt, Win, W0, seed, kw, device=device)
        except (RuntimeError, ValueError):
            gt[s] = {"test": float("inf"), "validation": float("inf"), "pas retenu": -1}
        save(res, path)


# --------------------------------------------------------------------------------------------- phase 1 : calibrage
def calibration_rule(cal):
    """(a*, a2, recalibrage, diagnostic) à partir des résultats de calibrage {a: {tâche: cellule}}."""
    M = m48()
    diag, margins, ok = {}, {}, []
    for a in GRID:
        cells = cal.get(str(a), {})
        if not all(t in cells and len(cells[t].get("structures", {})) == len(M.NAMES) for t in PLANTED):
            continue
        fail_i = fail_ii = 0
        mg = []
        for t in PLANTED:
            c = cells[t]
            g = {s: v["test"] for s, v in c["structures"].items()}
            exp = M.EXPECTED[t]
            margin = min(v for s, v in g.items() if s != exp) - g[exp]           # > 0 : la plantée gagne
            mg.append(margin)
            fail_i += not (c["classique"]["test"] - c["oracle"]["test"] > 0.5)
            fail_ii += not (margin >= 0.1)
        diag[a] = {"(i) réservoir figé trop près du plancher": fail_i, "(ii) plantée ne gagne pas de 0,1": fail_ii}
        margins[a] = float(np.mean(mg))
        if fail_i == 0 and fail_ii == 0:
            ok.append(a)
    if len(diag) < len(GRID):
        return None
    if ok:
        a_star, recal = min(ok), False
    else:
        a_star, recal = max(margins, key=margins.get), True
    return {"a*": a_star, "a2": NEXT[a_star], "recalibrage": recal, "diagnostic": diag, "avance moyenne": margins}


def load_calibration():
    """Fusionne les fichiers de calibrage (un par amplitude) : {a: {tâche: cellule}}."""
    cal = {}
    for f in sorted(glob.glob(str(core.out_path("data", "preenr_calibrage_a*.json")))):
        cal.update(json.load(open(f))["calibrage"])
    return cal


def run_calibration(a):
    import torch
    torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    t0 = time.time()
    for amp in (a.amps or GRID):
        path = core.out_path("data", f"preenr_calibrage_a{amp}.json")
        res = json.load(open(path)) if path.exists() else {"graine": CAL_SEED, "calibrage": {}}
        for task in PLANTED:
            cell = res["calibrage"].setdefault(str(amp), {}).setdefault(task, {})
            ground_truth(task, amp, CAL_SEED, device, cell, path, res)
            print(f"[{(time.time() - t0) / 60:5.1f} min] calibrage a = {amp} {task}", flush=True)
    rule = calibration_rule(load_calibration())
    print("règle :", json.dumps(rule, ensure_ascii=False) if rule else "les trois amplitudes ne sont pas encore toutes calculées")


# --------------------------------------------------------------------------------------------- phase 2 : P2
def henrici(W):
    ev = np.linalg.eigvals(W)
    f2 = (W ** 2).sum()
    return float(np.sqrt(max(f2 - (np.abs(ev) ** 2).sum(), 0.0)) / np.sqrt(f2))


def run_p2(a, path):
    import torch
    from hfrnn import structure_signals as SS
    from hfrnn import torch_models as TM
    torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    m46, m47 = _load("46_sous_espace_appris.py", "m46"), _load("47_protocole_lie.py", "m47")
    res = json.load(open(path)) if path.exists() else {"graine": a.seed, "p2": {}}
    t0 = time.time()
    for task in REAL:
        cell = res["p2"].setdefault(task, {})
        if all(k in cell for k in ("premières", "aléatoires", "bande k = 10", "post-hoc")):
            continue
        u, y, nt = m46.data(task)
        d = 1 if u.ndim == 1 else u.shape[1]
        Win, W = SS.reservoir(100, d, a.seed)
        w, c = SS.ridge_readout(u, y, nt, Win, W)
        rng = np.random.default_rng(900 + a.seed)                       # mêmes tirages que l'étape 4
        starts = rng.integers(0, int(0.75 * nt) - core.WIN_LEN - 1, m47.N_A)
        G = SS.window_grads(u, y, starts, Win, W, w, c)[0].astype(np.float32)
        _, s, Vt = np.linalg.svd(G, full_matrices=False)
        ev = s ** 2 / len(G)
        top = Vt[:1000]
        Q = np.linalg.qr(np.random.default_rng(1000 + a.seed).normal(size=(Vt.shape[1], 1000)))[0].T
        for key, kw in (("premières", {"kind": "subspace", "V": top}), ("aléatoires", {"kind": "subspace", "V": Q}),
                        ("bande k = 10", {"kind": "decomp", "form": "iwasawa_N", "band": 10})):
            if key in cell:
                continue
            v, _ = TM.train_regression(u, y, Win, W, a.seed, steps=core.STEPS, device=device, n_train=nt,
                                       **{k: x for k, x in kw.items()})
            cell[key] = float(np.log10(v))
            save(res, path)
        if "post-hoc" not in cell:                                        # analyse descriptive, assumée post-hoc
            _, info = TM.train_regression(u, y, Win, W, a.seed, kind="convex", alpha=1.0, steps=core.STEPS, device=device,
                                          n_train=nt, return_model=True)
            Wt = info["model"].weights()[1].detach().cpu().double().numpy()
            dW = (Wt - W).ravel()
            part = float(((top.astype(np.float64) @ dW) ** 2).sum() / (dW ** 2).sum())
            sv = np.linalg.svd((Wt - W), compute_uv=False)
            cell["post-hoc"] = {"RNN entraîné": float(np.log10(info.get("checkpoints", {}).get(core.STEPS, np.nan))),
                                "part de ΔW dans les 1 000 premières": part, "hasard": 1000 / dW.size,
                                "rang stable de ΔW": float((sv ** 2).sum() / sv[0] ** 2),
                                "Henrici W₀": henrici(W), "Henrici W₀ + ΔW": henrici(Wt),
                                "ratio de participation de C": float(ev.sum() ** 2 / (ev ** 2).sum())}
            save(res, path)
        del G, Vt
        gc.collect()
        print(f"[{(time.time() - t0) / 60:5.1f} min] P2 {task} graine {a.seed}", flush=True)


# --------------------------------------------------------------------------------------------- phase 3 : P1
def run_p1(a, path):
    import torch
    from hfrnn import structure_signals as SS
    torch.set_num_threads(a.threads)
    device = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    cal = load_calibration()
    rule = calibration_rule(cal)
    if not rule:
        M_ = m48()
        state = []
        for amp in GRID:
            done = sum(len(cal.get(str(amp), {}).get(t, {}).get("structures", {})) == len(M_.NAMES) for t in PLANTED)
            state.append(f"a = {amp} : {done}/4 tâches")
        raise SystemExit("la phase 1 (calibrage) doit être terminée avant la phase 3 — état dans cette session : "
                         + " ; ".join(state) + ". Lancer d'abord le bloc de calibrage (ou fusionner son zip s'il a tourné "
                         "dans une autre session).")
    amps = [rule["a*"], rule["a2"]]
    M = m48()
    res = json.load(open(path)) if path.exists() else {"graine": a.seed, "amplitudes": amps, "p1": {}, "bases": {}}
    tasks = PLANTED + ["T5 témoin"]
    if a.part == "vérité":                                               # vérité terrain : fichier à part, sans bases
        t0 = time.time()
        for amp in amps:
            if a.amps and amp not in a.amps:                               # découpage en blocs courts (Colab)
                continue
            for task in tasks:
                if a.tasks and not any(task.startswith(x) for x in a.tasks):
                    continue
                cell = res["p1"].setdefault(f"{task} | a = {amp}", {})
                ground_truth(task, amp, a.seed, device, cell, path, res)
                print(f"[{(time.time() - t0) / 60:5.1f} min] P1 vérité terrain {task} a = {amp} graine {a.seed}", flush=True)
        return
    Win, W0 = SS.reservoir(100, 1, a.seed)
    S = M.structures()
    bases = None
    t0 = time.time()
    for amp in amps:                                                     # signaux
        for task in tasks:
            cell = res["p1"].setdefault(f"{task} | a = {amp}", {})
            if "signaux" in cell:
                continue
            if bases is None:
                bases = {}
                for s, kw in S:
                    J = SS.tangent_jacobian(Win, W0, a.seed, kw)
                    bases[s] = SS.orthonormal_basis(J)
                    n_par = 0 if J is None or isinstance(J, str) else J.shape[1]
                    res["bases"].setdefault(s, M.check_basis(bases[s], n_par, a.seed))
                    del J
                    gc.collect()
            ok = {s: P for s, P in bases.items() if res["bases"][s]["ok"]}
            u, y, nt, _ = M.make_data(task, amp, Win, W0, a.seed)
            cell["signaux"] = M.gradient_signals(u, y, nt, Win, W0, a.seed, ok)
            cell["pca"] = M.pca_alignment(u, y, nt, Win, W0, a.seed, ok)
            save(res, path)
            print(f"[{(time.time() - t0) / 60:5.1f} min] P1 signaux {task} a = {amp} graine {a.seed}", flush=True)


# --------------------------------------------------------------------------------------------- rapport
def report(a):
    M = m48()
    out = {}
    print("=" * 100 + "\nPHASE 1 — CALIBRAGE (graine 3)")
    cal = load_calibration()
    if cal:
        for amp in GRID:
            for t in PLANTED:
                c = cal.get(str(amp), {}).get(t)
                if not c or len(c.get("structures", {})) < len(M.NAMES):
                    continue
                g = {s: v["test"] for s, v in c["structures"].items()}
                exp = M.EXPECTED[t]
                best_other = min((v, s) for s, v in g.items() if s != exp)
                print(f"  a = {amp} {t:14s} oracle {c['oracle']['test']:+.2f} classique {c['classique']['test']:+.2f} "
                      f"plantée {M.SHORT[exp]} {g[exp]:+.2f} ; meilleure autre {M.SHORT[best_other[1]]} {best_other[0]:+.2f} "
                      f"(avance {best_other[0] - g[exp]:+.2f}) ; pas retenus : "
                      + ", ".join(f"{M.SHORT[s]} {v['pas retenu']}" for s, v in c["structures"].items()))
        rule = calibration_rule(cal)
        out["calibrage"] = rule
        if rule:
            print(f"  diagnostic (nombre de tâches qui ratent chaque condition) : {rule['diagnostic']}")
            print(f"  → a* = {rule['a*']}, seconde amplitude {rule['a2']}"
                  + (" — RECALIBRAGE (aucun a ne satisfait la règle ; issue de secours)" if rule["recalibrage"] else ""))
    print("=" * 100 + "\nPHASE 2 — P2 (graines 3, 4, 5 ; log10 NMSE de test)")
    P2 = {}
    for f in sorted(glob.glob(str(core.out_path("data", "preenr_p2_s[0-9]*.json")))):
        r = json.load(open(f))
        for t, c in r["p2"].items():
            if all(k in c for k in ("premières", "aléatoires", "bande k = 10")):
                P2.setdefault(t, {})[str(r["graine"])] = c
    if P2:
        conds = {}
        for t in REAL:
            if t not in P2:
                continue
            seeds = sorted(P2[t])
            mean = {k: float(np.mean([P2[t][s][k] for s in seeds])) for k in ("premières", "aléatoires", "bande k = 10")}
            d_b, d_a = mean["premières"] - mean["bande k = 10"], mean["premières"] - mean["aléatoires"]
            print(f"  {t:20s} graines {seeds} : premières {mean['premières']:+.2f}, aléatoires {mean['aléatoires']:+.2f}, "
                  f"bande {mean['bande k = 10']:+.2f} ; premières − bande {d_b:+.2f}, premières − aléatoires {d_a:+.2f}"
                  f" ; par graine (− bande) " + ", ".join(f"{P2[t][s]['premières'] - P2[t][s]['bande k = 10']:+.2f}" for s in seeds))
            if t in BAND_LOSES:
                conds[f"{t} : premières < bande"] = d_b < 0
            else:
                conds[f"{t} : premières > aléatoires"] = d_a > 0
        complete = len(conds) == 5 and all(len(P2[t]) == 3 for t in REAL)
        verdict = ("PASSE" if all(conds.values()) else "ÉCHOUE") if complete else "incomplet"
        print("  conditions : " + " ; ".join(f"{k} {'oui' if v else 'NON'}" for k, v in conds.items()))
        print(f"  → P2 {verdict} ({sum(conds.values())}/5 conditions)")
        out["P2"] = {"conditions": conds, "verdict": verdict}
        print("  analyse post-hoc (descriptive, assumée comme telle ; moyenne sur les graines) :")
        for t in REAL:
            ph = [P2[t][s]["post-hoc"] for s in P2.get(t, {}) if "post-hoc" in P2[t][s]]
            if ph:
                keys = [k for k in ph[0] if k != "hasard"]
                print(f"    {t:20s} " + " ; ".join(f"{k} {np.mean([p[k] for p in ph]):.3g}" for k in keys)
                      + f" (hasard {ph[0]['hasard']:.2f})")
        out["P2 post-hoc"] = {t: {k: float(np.mean([P2[t][s]["post-hoc"][k] for s in P2[t] if "post-hoc" in P2[t][s]]))
                                  for k in next(iter(P2[t].values())).get("post-hoc", {})} for t in P2}
    print("=" * 100 + "\nPHASE 3 — P1 (graines 4, 5, 6)")
    R, GT = {}, {}
    for f in sorted(glob.glob(str(core.out_path("data", "preenr_p1_s[0-9]*.json")))):
        r = json.load(open(f))
        for k, c in r["p1"].items():
            if "signaux" in c:
                R.setdefault(k, {})[str(r["graine"])] = c
    for f in sorted(glob.glob(str(core.out_path("data", "preenr_p1vt_s[0-9]*.json")))):
        r = json.load(open(f))
        for k, c in r["p1"].items():
            GT.setdefault(k, {})[str(r["graine"])] = c
    if R:
        amps = sorted({float(k.split("a = ")[1]) for k in R})
        sigs = M.all_signals()
        table = {}
        for sig in sigs:
            hits, cases = 0, 0
            row = {}
            for amp in amps:
                for t in PLANTED + ["T5 témoin"]:
                    k = f"{t} | a = {amp}"
                    if k not in R:
                        continue
                    per = [M.signal_values(R[k][s], sig) for s in sorted(R[k])]
                    names = [s for s in M.CHOICE if all(s in p for p in per)]
                    V = np.array([[p[s] for p in per] for s in names], float)
                    chosen = names[int(np.argmax(np.argsort(np.argsort(V, 0), 0).mean(1)))]
                    row[k] = M.SHORT[chosen]
                    if t != "T5 témoin":
                        cases += 1
                        hits += chosen == M.EXPECTED[t]
            table[sig] = {"trouvées": hits, "cas": cases, "choix": row}
        seeds_ok = all(len(v) == 3 for v in R.values()) and len(R) == 2 * 5
        p1 = table[P1_SIGNAL]
        verdict = ("PASSE" if p1["trouvées"] >= 5 else "ÉCHOUE") if seeds_ok else "incomplet"
        for sig, v in table.items():
            print(f"  {sig:52s} {v['trouvées']} / {v['cas']}" + ("   ← P1" if sig == P1_SIGNAL else ""))
        print(f"  → P1 {verdict} : {p1['trouvées']} / {p1['cas']} (seuil 5 / 8) ; choix : "
              + " ; ".join(f"{k} → {c}" for k, c in p1["choix"].items()))
        out["P1"] = {"verdict": verdict, "signaux": table}
        print("  vérité terrain des graines de test (le calibrage tient-il ?) :")
        for k in sorted(GT):
            cells = [GT[k][s] for s in GT[k] if len(GT[k][s].get("structures", {})) == len(M.NAMES)]
            if not cells:
                continue
            t = k.split(" | ")[0]
            g = {s: float(np.mean([c["structures"][s]["test"] for c in cells])) for s in M.NAMES}
            cl = float(np.mean([c["classique"]["test"] - c["oracle"]["test"] for c in cells]))
            exp = M.EXPECTED[t]
            margin = min(v for s, v in g.items() if s != exp) - g[exp]
            print(f"    {k:22s} ({len(cells)} graines) classique − oracle {cl:+.2f} ; meilleure {M.SHORT[min(g, key=g.get)]} ; "
                  f"avance de la plantée {margin:+.2f}")
    json.dump(out, open(core.out_path("data", "preenr_summary.json"), "w"), indent=1, ensure_ascii=False, default=str)
    figure(out)


def figure(out):
    if "P2" not in out and "P1" not in out:
        return
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    if "P1" in out:
        t = out["P1"]["signaux"]
        sigs = list(t)
        ax[0].barh(range(len(sigs)), [t[s]["trouvées"] for s in sigs],
                   color=["tab:red" if s == P1_SIGNAL else "tab:blue" for s in sigs])
        ax[0].axvline(5, color="tab:green", ls=":", label="seuil de P1 (5 / 8)")
        ax[0].axvline(8 / 6, color="gray", ls="--", label="hasard")
        ax[0].set_yticks(range(len(sigs))); ax[0].set_yticklabels(sigs, fontsize=7); ax[0].invert_yaxis()
        ax[0].set_xlabel("structure plantée retrouvée (cas sur 8)"); ax[0].legend(fontsize=7)
        ax[0].set_title(f"P1 : {out['P1']['verdict']}", fontsize=10)
    if "P2" in out:
        c = out["P2"]["conditions"]
        ax[1].barh(range(len(c)), [1 if v else 0 for v in c.values()], color=["tab:green" if v else "tab:red" for v in c.values()])
        ax[1].set_yticks(range(len(c))); ax[1].set_yticklabels(list(c), fontsize=8); ax[1].invert_yaxis()
        ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["non", "oui"])
        ax[1].set_title(f"P2 : {out['P2']['verdict']}", fontsize=10)
    fig.tight_layout(); fig.savefig(core.out_path("figures", "preenregistre.png"), dpi=110)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["calibrage", "p2", "p1", "rapport"], required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--amps", type=float, nargs="*", help="calibrage : amplitudes à calculer (défaut : toute la grille)")
    p.add_argument("--part", choices=["signaux", "vérité"], default="signaux", help="P1 : signaux ou vérité terrain")
    p.add_argument("--tasks", nargs="*", help="P1, vérité terrain : tâches à calculer (début du nom, ex. T1 T2)")
    a = p.parse_args()
    if a.phase == "calibrage":
        run_calibration(a)
    elif a.phase == "p2":
        if a.seed not in P2_SEEDS:
            raise SystemExit(f"P2 : graines {P2_SEEDS} seulement (pré-enregistré)")
        run_p2(a, core.out_path("data", f"preenr_p2_s{a.seed}.json"))
    elif a.phase == "p1":
        if a.seed not in P1_SEEDS:
            raise SystemExit(f"P1 : graines {P1_SEEDS} seulement (pré-enregistré)")
        name = "preenr_p1_s" if a.part == "signaux" else "preenr_p1vt_s"
        run_p1(a, core.out_path("data", f"{name}{a.seed}.json"))
    else:
        report(a)
