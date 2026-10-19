"""Parties ajoutées en dernier : réservoir accordé linéaire, entrées figées et reprise d'entraînement, plusieurs
tâches, recherche équitable, MNIST pixel par pixel, capacité mémoire, synthèse. Rapides, CPU, sans téléchargement."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hfrnn import core, tasks  # noqa: E402
from hfrnn import torch_models as TM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), ROOT / "experiments" / name)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def run_script(name, args, tmp_path, timeout=900):
    env = dict(os.environ, HFRNN_RESULTS_DIR=str(tmp_path), PYTHONPATH=str(ROOT / "src"))
    out = subprocess.run([sys.executable, str(ROOT / "experiments" / name)] + args, env=env, capture_output=True,
                         text=True, timeout=timeout)
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout


# ---------------------------------------------------------------- réservoir accordé linéaire (partie 5.4)
@pytest.mark.parametrize("n", [20, 100])
def test_linear_tuned_learns_only_eigenvalues_and_inputs(n):
    Win, W = TM.make_reservoir(n, seed=0)
    full = TM.build_model(Win, W, "lru", 0, torch.float64, "cpu")
    tuned = TM.build_model(Win, W, "linear_tuned", 0, torch.float64, "cpu")
    names = {k for k, p in tuned.named_parameters() if p.requires_grad}
    assert names == {f"layers.0.{k}" for k in ("nu_log", "theta_log", "gamma_log", "B_re", "B_im")}
    assert TM.n_trainable(tuned) == 5 * (n // 2)           # λ (2), γ, B (2) par mode, une entrée
    assert TM.n_trainable(full) > 20 * TM.n_trainable(tuned) or n < 50
    two = TM.build_model(Win, W, "linear_tuned", 0, torch.float64, "cpu", layers=2)
    assert TM.n_trainable(two) == 5 * (n // 2) + 3 * (n // 2)   # couche 2 : λ et γ seulement


def test_linear_tuned_starts_like_the_lru_and_trains():
    u, y = tasks.load("NARMA10")
    Win, W = TM.make_reservoir(30, seed=1)
    a = TM.build_model(Win, W, "lru", 1, torch.float64, "cpu")
    b = TM.build_model(Win, W, "linear_tuned", 1, torch.float64, "cpu")
    U = torch.as_tensor(u[:200], dtype=torch.float64).view(1, -1, 1)
    assert torch.allclose(a(U), b(U))                        # même point de départ : seuls les paramètres libres diffèrent
    s0 = TM.regression_nmse(b, torch.as_tensor(u, dtype=torch.float64), torch.as_tensor(y, dtype=torch.float64))
    s, info = TM.train_regression(u, y, Win, W, 1, kind="linear_tuned", steps=60, dtype=torch.float64, return_model=True)
    m = info["model"]
    assert np.isfinite(s) and s < s0
    assert torch.equal(m.layers[0].C_re, a.layers[0].C_re) and torch.equal(m.layers[0].D, a.layers[0].D)


# ---------------------------------------------------------------- entrées figées, reprise d'entraînement (7.3)
def test_frozen_inputs_stay_at_the_reservoir_inputs():
    u, y = tasks.load("Mémoire (délai 15)")
    Win, W = TM.make_reservoir(20, seed=0)
    s, info = TM.train_regression(u, y, Win, W, 0, kind="spectrum", steps=30, dtype=torch.float64, learn_win=False,
                                  return_model=True)
    m = info["model"]
    assert float(m.Win_l.detach().abs().max()) == 0.0 and float((m.g.detach() - 1).abs().max()) > 0
    assert info["n_params"] == 2 * 20                       # g et biais


def test_training_can_continue_on_an_existing_model():
    u, y = tasks.load("NARMA10")
    Win, W = TM.make_reservoir(20, seed=0)
    _, i1 = TM.train_regression(u, y, Win, W, 0, kind="convex", alpha=1.0, steps=20, dtype=torch.float64,
                                return_model=True)
    before = i1["model"].W_l.detach().clone()
    _, i2 = TM.train_regression(u, y, Win, W, 0, kind="convex", alpha=1.0, steps=20, dtype=torch.float64,
                                return_model=True, model=i1["model"])
    assert i2["model"] is i1["model"] and not torch.equal(before, i2["model"].W_l)


def test_multitask_end_to_end(tmp_path):
    out = run_script("25_multitask.py", ["--n", "16", "--seeds", "1", "--steps", "8", "--jobs", "1", "--cpu"], tmp_path)
    res = json.load(open(tmp_path / "data" / "multitask_n16.json"))["results"]["0"]
    for t in load("25_multitask.py").TASKS:
        # remettre le réglage rangé d'une tâche dans un modèle commun redonne exactement son score
        assert res["contrôle, réglage rangé remis en place"][t] == pytest.approx(
            res["accordé, entrées partagées"][t]["final"], rel=1e-9)
        assert len(res["RNN séquentiel"][t]["après chaque étape"]) >= 1
    assert "Figure" in out


# ---------------------------------------------------------------- recherche équitable (6.4)
def test_fair_search_sampling_and_validation_split():
    F = load("26_fair_search.py")
    for m in F.MODELS:
        for t in F.TASKS:
            a, b = F.sample(m, t, 3), F.sample(m, t, 3)
            assert a == b and set(a) == set(F.MODELS[m]["space"])
            assert set(F.MODELS[m]["default"]) == set(F.MODELS[m]["space"])
    u, y = tasks.load("NARMA10")
    uv, yv = F.validation_split("NARMA10")
    nt = core.n_train(u)
    assert len(uv) == nt and np.array_equal(uv, u[:nt])     # le test (après nt) n'est jamais vu
    assert core.n_train(uv) < len(uv)                      # la fin de la partie entraînement sert de validation


def test_fair_search_end_to_end(tmp_path):
    run_script("26_fair_search.py", ["--quick", "--tasks", "NARMA10"], tmp_path)
    res = json.load(open(tmp_path / "data" / "fair_search_n24_quick.json"))
    F = load("26_fair_search.py")
    for m in F.MODELS:
        assert set(res["search"]["NARMA10"][m]) == {"défaut", "0", "1"}
        assert len(res["final"]["NARMA10"][m]["seeds"]) == 2


# ---------------------------------------------------------------- MNIST pixel par pixel (8.3)
def test_pixel_mnist_end_to_end(tmp_path):
    run_script("27_pixel_mnist.py", ["--quick", "--synthetic"], tmp_path, timeout=1500)
    res = json.load(open(tmp_path / "data" / "pixel_mnist_quick.json"))
    P = load("27_pixel_mnist.py")
    assert set(res["runs"]) == set(P.MODELS)
    for m in P.MODELS:
        lr = P.best_lr(res, m)
        assert len(res["runs"][m][f"{lr:g}"]) == 2           # seed 0 + une autre seed au learning rate retenu


# ---------------------------------------------------------------- capacité mémoire (3.4)
def test_memory_capacity_of_a_linear_delay_line_is_exact():
    C = load("24_memory_capacity.py")
    u = C.probe_input(0)
    n = 12                                                    # registre à décalage : l'état garde les 12 dernières entrées
    X = np.stack([np.concatenate([np.zeros(k), u[:len(u) - k]]) for k in range(n)], 1)
    caps = C.summarize(C.capacities(X, u))
    assert caps["MC_k"][0] == pytest.approx(1.0, abs=1e-6)    # délai 1 … 11 parfaitement retrouvés
    assert caps["MC_k"][10] == pytest.approx(1.0, abs=1e-6)
    assert caps["MC_k"][20] == pytest.approx(0.0, abs=0.01)   # au-delà : rien
    assert caps["MC"] == pytest.approx(11, abs=0.2)
    assert caps["nonlin2"] < 0.5                              # un état linéaire ne calcule pas de carrés


# ---------------------------------------------------------------- synthèse (partie 9)
def test_synthesis_runs_on_the_repository_results(tmp_path):
    import shutil
    shutil.copytree(ROOT / "results" / "data", tmp_path / "data")
    run_script("28_synthesis.py", [], tmp_path, timeout=600)
    s = json.load(open(tmp_path / "data" / "synthesis.json"))
    assert set(s) == {"Mackey-Glass (+15)", "NARMA10", "Mémoire (délai 15)"}
    for slug in ("mackey_glass", "narma10", "memoire", "pareto", "beyond"):
        assert (tmp_path / "figures" / f"synthesis_{slug}.png").exists()
