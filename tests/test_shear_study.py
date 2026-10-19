"""Le Colab de la partie 5 : les listes de modèles sont cohérentes, et la recherche équitable reprend les anciens tirages."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

EXP = Path(__file__).resolve().parents[1] / "experiments"


def load(fname, alias):
    spec = importlib.util.spec_from_file_location(alias, EXP / fname)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def S():
    return load("43_shear_study.py", "shear_study")


@pytest.fixture(scope="module")
def F():
    return load("26_fair_search.py", "fair26")


def test_labels_unique_per_mode(S):
    for mode, (models, tasks_, n) in S.MODES.items():
        labels = [l for l, _ in models]
        assert len(labels) == len(set(labels)), mode
        assert n in (200, 500)
    labels = [l for l, _ in S.REAL]
    assert len(labels) == len(set(labels))


def test_budget_labels_do_not_clash_with_existing(S):
    old = json.load(open(EXP.parent / "results" / "data" / "budget_inputs.json"))["results"]
    present = set().union(*[set(c) for c in old.values()])
    for l, _ in S.BUDGET + S.GATED:
        assert l not in present, l                              # sinon le Colab ne calculerait rien
    for l in S.BUDGET_REFS + S.GATED_REFS:
        assert l in present, l                                  # chaque référence existe déjà (graines 0 à 2)


def test_gated_variants_are_valid(S):
    torch = pytest.importorskip("torch")
    from hfrnn import torch_models as TM
    Win, W = TM.make_reservoir(16, n_in=1, rho=0.9, in_scale=0.5, sparsity=0.2, seed=0)
    for l, kw in S.GATED + S.FRESH[-5:-3] + S.REAL[-2:]:
        kw = dict(kw)
        kw.pop("n", None)
        kind = kw.pop("kind")
        m = TM.build_model(Win, W, kind, 0, torch.float32, "cpu", **kw)
        assert m(torch.zeros(1, 5, 1)).shape == (1, 5, 16), l


def test_fresh_data_differs_from_old(S):
    for name in S.SIX:
        u0, y0, nt0 = S.data(name)
        u1, y1, nt1 = S.fresh_data(name)
        assert len(u1) == len(u0) == 4000 and nt1 == nt0 == 3000, name
        assert not np.allclose(u0, u1), name


def test_old_fair_search_trials_are_unchanged(F):
    d = json.load(open(EXP.parent / "results" / "data" / "fair_search_n500.json"))
    n = 0
    for t, D in d["search"].items():
        for m, S_ in D.items():
            for i, v in S_.items():
                if i == "défaut":
                    continue
                new = F.sample(m, t, int(i))
                assert new == pytest.approx(v["cfg"], rel=1e-9), (t, m, i)
                n += 1
    assert n > 100


def test_new_fair_search_models_sample(F):
    for m in list(F.MODELS)[-4:]:
        cfg = F.sample(m, "Lorenz (+10)", 0)
        assert set(cfg) == set(F.MODELS[m]["space"])
        assert cfg == F.sample(m, "Lorenz (+10)", 0)


def test_report_ignores_summary_files(S, tmp_path, monkeypatch):
    """shear_<mode>_summary.json ne doit pas être pris pour un fichier de graine (KeyError: 'results' sur Colab)."""
    monkeypatch.setattr(S.core, "RESULTS_DIR", tmp_path)
    d = tmp_path / "data"
    d.mkdir()
    (d / "shear_curve_s0.json").write_text(json.dumps({"args": {}, "results": {"NARMA10": {"accordé": {"0": {"nmse": 0.1, "train": 0.1, "n_params": 5}}}}}))
    (d / "shear_curve_summary.json").write_text(json.dumps({"NARMA10": {}}))
    R = S.load_results("curve")
    assert R["NARMA10"]["accordé"]["0"]["nmse"] == 0.1


def test_diverging_trial_is_infinite_not_a_crash(F, monkeypatch):
    pytest.importorskip("torch")
    from hfrnn import torch_models as TM

    def boom(*a, **k):
        raise RuntimeError("torch.linalg.solve: The solver failed because the input matrix is singular.")
    monkeypatch.setattr(TM, "train_regression", boom)
    u = np.sin(np.arange(300) * 0.1)
    v, npar = F.fit(u, u, "cisaillé linéaire à porte d'entrée (k cherché)",
                    F.MODELS["cisaillé linéaire à porte d'entrée (k cherché)"]["default"], 0, 16, 2, "cpu")
    assert v == float("inf") and npar > 0
