"""Parties 8.5 à 8.7 : carte des tâches, benchmarks d'identification, archive UEA ; entrées multiples et nombre de pas
d'entraînement choisi dans train_regression ; classifieur à K classes. Rapides, CPU, sans téléchargement."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hfrnn import core, task_map, tasks  # noqa: E402
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


# ---------------------------------------------------------------- générateurs de la carte des tâches
@pytest.mark.parametrize("name", list(task_map.TASKS))
def test_task_map_generators(name):
    u, y, nt = task_map.load(name)
    assert len(u) == len(y) and len(u) - nt == task_map.T_TEST
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(y))
    assert abs(y[:nt].mean()) < 1e-6 and abs(y[:nt].std() - 1) < 1e-6        # normalisé sur l'entraînement
    u2, y2, _ = task_map.load(name)
    assert np.array_equal(u, u2) and np.array_equal(y, y2)                      # reproductible
    _, fam = task_map.TASKS[name]
    assert isinstance(fam, str) and fam


def test_channel_equalization_target_is_a_past_symbol():
    u, y, nt = task_map.channel_equalization()
    assert len(np.unique(np.round(y, 6))) == 4                                   # 4 symboles (normalisés)


def test_linear_baseline_solves_a_pure_delay_and_not_chaos():
    u, y, nt = task_map.long_delay(30)
    e, info = task_map.linear_baseline(u, y, nt)
    assert e < 1e-8 and info["lags"] >= 31
    u, y, nt = task_map.henon()
    e, _ = task_map.linear_baseline(u, y, nt)
    assert e > 0.3                                                                # Hénon est non linéaire


def test_rayleigh_channel_is_almost_linear():
    u, y, nt = task_map.rayleigh_csi()
    assert u.shape == (4000, 2)
    e, _ = task_map.linear_baseline(u, y, nt)
    assert e < 0.05                                                               # le contrôle : un linéaire suffit


# ---------------------------------------------------------------- train_regression : entrées multiples, n_train
def test_train_regression_with_vector_input_and_custom_split():
    u, y, nt = task_map.power_amplifier()
    Win, W = TM.make_reservoir(20, n_in=2, seed=0)
    for kind in ("spectrum", "gru", "linear_tuned", "reservoir"):
        s, info = TM.train_regression(u, y, Win, W, 0, kind=kind, steps=5, n_train=nt, dtype=torch.float64)
        assert np.isfinite(s)
    # n_train explicite : le score ne regarde que la fin
    u1, y1 = tasks.load("NARMA10")
    m = TM.build_model(*TM.make_reservoir(20, seed=0), "reservoir", 0, torch.float64, "cpu")
    ut, yt = torch.as_tensor(u1, dtype=torch.float64), torch.as_tensor(y1, dtype=torch.float64)
    assert TM.regression_nmse(m, ut, yt) == pytest.approx(TM.regression_nmse(m, ut, yt, n_train=3000))
    assert TM.regression_nmse(m, ut, yt, n_train=2000) != pytest.approx(TM.regression_nmse(m, ut, yt))


def test_scalar_input_unchanged_by_the_generalization():
    """Entrée scalaire : as_batch donne exactement l'ancien format (1, T, 1)."""
    ut = torch.arange(5.0)
    assert TM.as_batch(ut).shape == (1, 5, 1)
    assert TM.as_batch(torch.zeros(5, 3)).shape == (1, 5, 3)


# ---------------------------------------------------------------- classifieur à K classes
def test_fit_classifier_with_three_classes():
    r = np.random.default_rng(0)
    X = torch.as_tensor(r.normal(size=(60, 10, 2)), dtype=torch.float32)
    y = torch.as_tensor(r.integers(0, 3, 60))
    data = {"Xtr": X[:40], "ytr": y[:40], "Xva": X[40:], "yva": y[40:], "Xte": X[40:], "yte": y[40:]}
    out = TM.fit_classifier(data, "spectrum", 0, n=8, epochs=2, batch=8, test=True)
    assert len(out["val"]) == 2 and 0 <= out["test"][-1] <= 1


# ---------------------------------------------------------------- scripts de bout en bout (versions rapides)
def test_task_map_end_to_end(tmp_path):
    out = run_script("29_task_map.py", ["--quick"], tmp_path)
    assert "victoires de l'accordé" in out
    s = json.load(open(tmp_path / "data" / "task_map_n20_summary.json"))
    assert len(s["tâches"]) == 3
    assert (tmp_path / "figures" / "task_map_n20.png").exists()


def test_nonlinear_benchmarks_end_to_end(tmp_path):
    out = run_script("30_nonlinear_benchmarks.py", ["--synthetic", "--quick"], tmp_path)
    assert "régression linéaire" in out
    s = json.load(open(tmp_path / "data" / "nonlinear_benchmarks_quick_summary.json"))
    assert len(s) == 2 and all("réservoir accordé" in v["rmse"] for v in s.values())


def test_nonlinear_benchmark_evaluation_starts_from_zero_state():
    """Le test est simulé depuis un état nul : deux tests identiques donnent la même erreur, quelle que soit la fin
    de l'entraînement."""
    m = load("30_nonlinear_benchmarks.py")
    u, y, _ = task_map.wiener_hammerstein(T=1200, seed=1)
    Win, W = TM.make_reservoir(30, seed=0)
    model = TM.build_model(Win, W, "reservoir", 0, torch.float32, "cpu")
    ut, yt = torch.as_tensor(u[:800], dtype=torch.float32), torch.as_tensor(y[:800], dtype=torch.float32)
    r = m.evaluate(model, ut, yt, {"a": (u[800:], y[800:], 50), "b": (u[800:], y[800:], 50)}, 1.0, "cpu")
    assert r["a"] == r["b"] and np.isfinite(r["a"]["rmse"])


def test_uea_padding_and_split():
    m = load("31_uea.py")
    X = [np.ones((2, 3)), np.ones((2, 5))]
    P = m.pad(X)
    assert P.shape == (2, 5, 2) and P[0, :2].sum() == 0 and P[0, 2:].sum() == 6     # zéros au début
    y = np.array([0] * 10 + [1] * 5)
    tr, va = m.split_val(y)
    assert len(set(tr) & set(va)) == 0 and len(tr) + len(va) == 15
    assert set(y[va]) == {0, 1}


def test_uea_end_to_end(tmp_path):
    out = run_script("31_uea.py", ["--quick", "--synthetic"], tmp_path)
    assert "rang moyen" in out
    s = json.load(open(tmp_path / "data" / "uea_n16_summary.json"))
    assert len(s["jeux"]) == 2


def test_multitask_readout_on_device_of_states():
    """Régression du bug GPU du bloc « plusieurs tâches » : nmse_with crée ses tenseurs sur l'appareil des états."""
    src = (ROOT / "experiments" / "25_multitask.py").read_text()
    assert "torch.ones(len(X), 1, dtype=X.dtype, device=X.device)" in src


# ---------------------------------------------------------------- deuxième lot : portes sur la carte, noms des tests
def test_gated_models_in_task_map():
    m = load("29_task_map.py")
    labels = [l for l, _, _ in m.MODELS]
    assert m.GATE in labels and "accordé sélectif" in labels and len(labels) == len(set(labels))


def test_silverbox_tests_get_distinct_names():
    """Le paquet nonlinear-benchmarks nomme deux tests de Silverbox « test SB multisine » : on les distingue."""
    m = load("30_nonlinear_benchmarks.py")

    class T:
        name = "test SB multisine"
    names = [m.test_name("Silverbox", T(), i) for i in range(3)]
    assert len(set(names)) == 3
    assert m.test_name("Cascaded Tanks", T(), 0) == "test SB multisine"


def test_linear_gallery_helpers():
    m = load("32_linear_gallery.py")
    Win, W = TM.make_reservoir(10, seed=0, sparsity=0.3)
    for kind in ("lru", "linear_tuned", "spectrum"):
        model = TM.build_model(Win, W, kind, 0, torch.float64, "cpu")
        lam = m.eig(model, kind)
        assert np.all(np.abs(lam) < 1.0 + 1e-9) and len(lam) in (5, 10)
        g = m.groups(model, kind)
        tr = m.trainable(model, kind)
        assert set(g) == set(tr)
    tr = m.trainable(TM.build_model(Win, W, "linear_tuned", 0, torch.float64, "cpu"), "linear_tuned")
    assert not tr["C (sorties)"] and not tr["D (direct)"] and tr["λ (module, phase)"]


def test_amplifier_control_end_to_end(tmp_path):
    out = run_script("33_amplifier_control.py", ["--quick"], tmp_path)
    s = json.load(open(tmp_path / "data" / "amplifier_control_summary.json"))
    assert "réservoir accordé" in s and set(s["réservoir accordé"]) == {"5", "10"}


# ---------------------------------------------------------------- troisième lot : cuDNN, GMP, amplificateur
def test_gru_chunked_forward_matches_single_call():
    """Le GRU découpé en morceaux (contournement de CUDNN_STATUS_NOT_SUPPORTED) donne les mêmes états."""
    from hfrnn.torch_baselines import Baseline
    b = Baseline("gru", 12, n_in=1, seed=0, dtype=torch.float64)
    U = torch.randn(1, 50, 1, dtype=torch.float64)
    full = b(U)
    b.CHUNK = 7
    assert torch.allclose(b(U), full, atol=1e-12)
    l = Baseline("lstm", 12, n_in=2, seed=0, dtype=torch.float64)
    U2 = torch.randn(1, 30, 2, dtype=torch.float64)
    full = l(U2)
    l.CHUNK = 8
    assert torch.allclose(l(U2), full, atol=1e-12)


def test_regression_nmse_builds_no_graph():
    """Plus d'avertissement « requires_grad » : l'évaluation intermédiaire ne garde pas de graphe."""
    import warnings
    m = TM.build_model(*TM.make_reservoir(30, seed=0, sparsity=0.3), "spectrum", 0, torch.float64, "cpu")
    u1, y1 = tasks.load("NARMA10")
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*requires_grad.*")
        s = TM.regression_nmse(m, torch.as_tensor(u1[:600]), torch.as_tensor(y1[:600]), n_train=400)
    assert np.isfinite(s)


def test_memory_polynomial_fits_a_memoryless_saleh_amplifier():
    r = np.random.default_rng(0)
    x = (r.normal(size=3000) + 1j * r.normal(size=3000)) * 0.5
    a = np.abs(x)
    z = 2.1587 * a / (1 + 1.1517 * a ** 2) * np.exp(1j * (np.angle(x) + 4.0033 * a ** 2 / (1 + 9.104 * a ** 2)))
    u = np.stack([x.real, x.imag], 1)
    e, info = task_map.memory_polynomial_baseline(u, z.real, 2000)
    assert e < 1e-2 and info["K"] >= 5
    e_lin, _ = task_map.linear_baseline(u, z.real, 2000)
    assert e < e_lin / 5                                        # bien meilleur que la fenêtre linéaire


def test_amplifier_deep_dive_end_to_end(tmp_path):
    out = run_script("34_amplifier_deep_dive.py", ["--quick"], tmp_path)
    s = json.load(open(tmp_path / "data" / "amplifier_deep_dive_quick_summary.json"))
    assert set(s["tailles"]) == {"600", "2000"}
    assert "polynôme à mémoire généralisé (GMP)" in s["tailles"]["600"]
    assert all("entraînement" in s["tailles"]["600"][k] for k in ("réservoir accordé", "GRU, init soignée, lr 2e-3"))
    assert "GRU, entrées enrichies" in s["tailles"]["600"]
