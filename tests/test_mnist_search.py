"""Recherche d'hyperparamètres sur MNIST (partie 8.2) : moteur d'entraînement et protocole, sur données synthétiques.

Ignorés si PyTorch n'est pas installé. Aucun téléchargement de MNIST.
"""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from hfrnn import torch_models as TM  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mnist_search", ROOT / "experiments" / "23_mnist_search.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


@pytest.fixture(scope="module")
def data():
    return S.synthetic_data(800, 200, 200)


@pytest.mark.parametrize("model", list(S.MODELS))
def test_sampling_is_deterministic_and_inside_the_space(model):
    for i in range(30):
        a, b = S.sample(model, i), S.sample(model, i)
        assert a == b
        assert set(a) == set(S.MODELS[model]["space"])
        for k, spec in S.MODELS[model]["space"].items():
            if spec[0] in ("log", "uni"):
                assert spec[1] <= a[k] <= spec[2]
            else:
                assert a[k] in spec[1]
    assert S.sample(model, 0) != S.sample(model, 1)


def test_models_get_independent_draws():
    # même nombre d'essais, mais tirages différents d'un modèle à l'autre
    assert S.sample("RNN", 3)["lr"] != S.sample("GRU", 3)["lr"]


def test_default_configs_cover_their_search_space():
    for m, spec in S.MODELS.items():
        assert set(spec["default"]) == set(spec["space"])


@pytest.mark.parametrize("model", list(S.MODELS))
def test_fit_learns_synthetic_digits(data, model):
    r = S.fit(data, model, S.MODELS[model]["default"] | {"lr": 3e-3}, 0, epochs=3, n=24, test=True, device="cpu", batch=64)
    assert len(r["val"]) == 3 and len(r["test"]) == 3 and not r["diverged"]
    assert r["val"][-1] > 0.3 and r["test"][-1] > 0.3          # le hasard est à 0.1
    assert r["n_params"] > 0


def test_test_set_is_untouched_during_search(data):
    r = S.fit(data, "RNN", S.MODELS["RNN"]["default"], 0, epochs=1, n=16, test=False, device="cpu", batch=64)
    assert r["test"] is None


def test_fit_is_reproducible(data):
    kw = dict(epochs=1, n=16, test=False, device="cpu", batch=64)
    a = S.fit(data, "réservoir accordé", S.MODELS["réservoir accordé"]["default"], 3, **kw)
    b = S.fit(data, "réservoir accordé", S.MODELS["réservoir accordé"]["default"], 3, **kw)
    assert a["val"] == b["val"]


def test_diverged_run_counts_as_chance_and_does_not_crash(data):
    cfg = S.MODELS["RNN"]["default"] | {"lr": float("nan")}
    r = S.fit(data, "RNN", cfg, 0, epochs=3, n=16, test=True, device="cpu", batch=64)
    assert r["diverged"] and r["val"] == [0.1] * 3 and r["test"] == [0.1] * 3


def test_accordé_only_learns_amplitudes_inputs_and_bias(data):
    from hfrnn import torch_models as T
    Win, W = T.make_reservoir(32, n_in=28, seed=0)
    m = T.build_model(Win, W, "spectrum", 0, torch.float32, "cpu")
    names = {n for n, p in m.named_parameters() if p.requires_grad}
    assert names == {"g", "Win_l", "bias"}


def test_best_trial_ignores_the_default_and_breaks_ties_by_order():
    search = {"default": {"val": [0.99]}, "0": {"val": [0.90]}, "1": {"val": [0.95]}, "2": {"val": [0.95]}}
    assert S.best_trial(search) == "1"


def test_quick_pipeline_end_to_end_and_resumable(tmp_path):
    env = dict(os.environ, HFRNN_RESULTS_DIR=str(tmp_path), PYTHONPATH=str(ROOT / "src"))
    cmd = [sys.executable, str(ROOT / "experiments" / "23_mnist_search.py"), "--quick", "--synthetic"]
    out = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert out.returncode == 0, out.stderr[-2000:]
    path = tmp_path / "data" / "mnist_search_quick.json"
    res = json.load(open(path))
    assert set(res["search"]) == set(S.MODELS) and set(res["best"]) == set(S.MODELS)
    for m in S.MODELS:
        assert len(res["search"][m]) == 3                       # 2 essais + la configuration par défaut
        assert "recherche" in res["final"]["32"][m] and "défaut" in res["final"]["32"][m]
        assert "défaut" not in res["final"]["48"][m]            # pas de défaut aux tailles transférées
    assert (tmp_path / "figures" / "mnist_search_quick.png").exists()
    # relance : tout est repris, rien n'est recalculé
    out2 = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    assert out2.returncode == 0 and "recherche" not in out2.stdout.split("Recherche d'hyperparamètres")[0]
    assert json.load(open(path))["search"] == res["search"]
