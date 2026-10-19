[Contents](../README.en.md#the-report) · [Français](../rapport/reproduire.md)

# Reproducing

## Installation and check

```bash
pip install -e ".[dev]"          # numpy, scipy, matplotlib, pytest  (".[dev,gpu]" for PyTorch)
pytest                           # a few minutes; PyTorch tests are skipped without PyTorch
```

The experiments at 500 neurons and above run on GPU (Google Colab or Kaggle) with
[`notebooks/colab_gpu.ipynb`](../notebooks/colab_gpu.ipynb), which contains the measurements still pending (see the
[appendix](annexe_pistes.md)). Results already computed are in `results/data/`: each script picks up what exists and only
computes what is missing. All numpy runs are kept in `results/data/cache.json`; rerunning a script already computed is
instantaneous. For a quick trial without cache:
`HFRNN_STEPS=20 HFRNN_RESULTS_DIR=/tmp/essai python experiments/03_learned_alpha.py`.

Script names, arguments, model names inside the result files and figure labels are in French.

## Figures without computation

```bash
python experiments/38_schemas.py          # all the teaching diagrams (a few seconds)
python experiments/42_figures_rapport.py  # family assessment, who learns what, equal budget, fair search
python experiments/28_synthesis.py        # per-task syntheses and error/parameter front (part 8)
python experiments/11_model_gallery.py    # gallery: what each model does to the reservoir (~1 min)
python experiments/20_pitch_figure.py     # opening figure of part 8
python experiments/01_esn_demo.py         # demo: a reservoir on Fibonacci mod 10
```

## Experiments, part by part

Scripts are numbered in the order in which they were written; the list arranges them according to the plan of the report.

```bash
# Part 1 — mixing, braking (CPU)
python experiments/02_comparison.py        # 1.1 to 1.3, 1.7      (~45 min without cache)
python experiments/03_learned_alpha.py     # 1.4                  (~15 min)
python experiments/09_rotation.py          # 1.5 (spectra) and 3.1 (~20 min)
python experiments/04_robustness.py        # 1.6                  (~1 h)
# Part 2 — sculpted reservoir
python experiments/08_supermask.py         # 2.1 to 2.3 (CPU, ~30 min)
# Part 3 — rotated reservoir
python experiments/09_rotation.py          # 3.1 (CPU)
python experiments/36_decompositions.py --n 100 --seeds 3   # 3.2, 5.1, 6.1 (five tasks, 100 neurons)
# Part 4 — tuned reservoir
python experiments/05_gpu_regression.py --n 500 --seeds 3   # 4.1 and combinations of 4.4 (GPU)
python experiments/37_budget_and_inputs.py # 4.2 (who learns what) and 4.9 (equal budget) (GPU)
python experiments/40_truncated_tuned.py   # 4.2, truncated tuned reservoir (GPU)
python experiments/18_spectrum_controls.py # 4.2, spectrum controls (GPU)
python experiments/24_memory_capacity.py   # 4.3, memory capacity (CPU, ~10 min)
python experiments/13_exploration.py       # 4.4, combinations at 100 neurons (CPU, ~35 min)
python experiments/15_improvements.py --n 500   # 4.4, thirteen variants (GPU); 17_tuned_gallery.py: gallery
python experiments/21_gated.py             # 4.5 and 4.6, gates and linear recurrence (GPU)
python experiments/22_gated_gallery.py     # 4.5, gallery; 32_linear_gallery.py: 4.6, gallery
python experiments/16_scaling.py           # 4.7, from 100 to 2,000 neurons (GPU)
python experiments/19_fast_scaling.py      # 4.7, fast tuned reservoir up to 16,000 neurons (GPU)
python experiments/25_multitask.py         # 4.8 (CPU at 100 neurons; --n 500 on GPU)
# Parts 5 and 6 — sheared reservoir, nilpotent reservoir
python experiments/36_decompositions.py --n 100 --seeds 3   # 5.1 and 6.1; --n 500: 5.3 and 6.2 (full and banded)
python experiments/36b_decompositions_figure.py             # figure of 5.1
# Part 7 — couplings
python experiments/35_coupled_forms.py     # 7.2 (--n 100, CPU ~30 min) and 7.3 (--n 500, GPU)
python experiments/39_combinations.py      # 7.4 (GPU)
python experiments/41_readout_features.py  # 7.5 (GPU)
# Part 8 — synthesis, existing models, non-synthetic data
python experiments/12_comparisons.py       # 8.2, first protocol and gates at 4,000 steps (GPU; --report without GPU)
python experiments/26_fair_search.py       # 8.2, same search budget, sizes included (GPU; --quick: CPU test)
python experiments/06_gpu_smnist.py        # 8.3, row-by-row MNIST (GPU)
python experiments/23_mnist_search.py      # 8.3, MNIST with search (GPU)
python experiments/27_pixel_mnist.py       # 8.3 and 4.6, pixel-by-pixel MNIST (GPU)
python experiments/31_uea.py               # 8.3, UEA archive (GPU; pip install aeon)
python experiments/14_real_data.py         # 8.3, real series (CPU, ~15 min)
python experiments/29_task_map.py          # 8.3, map of tasks (GPU)
python experiments/33_amplifier_control.py # 8.3, amplifier, first control (GPU)
python experiments/34_amplifier_deep_dive.py   # 8.3, amplifier, overfitting and memory polynomial (GPU)
python experiments/30_nonlinear_benchmarks.py  # 8.3, real systems (GPU; pip install nonlinear-benchmarks;
                                               #      --extra: parts 6 and 7; --extra2: without g and sheared)
# Parts 5.4 to 5.10 — confirming the banded shear (one process per seed, several at once on a GPU)
python experiments/43_shear_study.py curve --seeds 0     # 5.4 and 5.5: k curve, controls (random positions, in W₀)
python experiments/43_shear_study.py budget --seeds 0    # 5.7: equal budget, with the results of budget_inputs.json
python experiments/43_shear_study.py gated --seeds 0     # 5.10: gated sheared reservoir at 200 neurons
python experiments/43_shear_study.py fresh --seeds 0     # 5.6: regenerated series, settings fixed in advance
python experiments/43_shear_study.py real --seeds 0      # 5.8: real systems
python experiments/43_shear_study.py freshreal --seeds 0 # 5.6: EMPS
python experiments/43_shear_study.py curve --report      # merges the per-seed files, tables and figure
python experiments/26_fair_search.py --tasks "Lorenz (+10)" --trials 8 --tag _lorenz --models "cisaillé"   # 5.9 and 8.2
# Part 9 — can the task choose its structure? (CPU, unless stated)
python experiments/36_decompositions.py --n 100 --seeds 3 --tag _p9a --only "en bande" "sans g"   # ground truth (~15 min)
python experiments/44_structure_signals.py   # 9.3: gradient signals at the start, evaluation (~30 min)
python experiments/45_signaux_dynamiques.py --seed 0   # 9.4: signals during training (one seed; then --report)
python experiments/46_sous_espace_appris.py --seed 0   # 9.5: subspaces taken from the gradient (one seed; then --report)
python experiments/47_protocole_lie.py --seed 0   # 9.5: PCA of the tangent gradient M⁻¹G, blocks and unfreezing (one seed; then --report)
python experiments/48_enseignants.py --seed 0     # 9.6: tasks with a planted structure (one seed; then --report)
python experiments/49_preenregistre.py --phase calibrage --amps 0.2   # 9.7: check on fresh seeds, see the script header
```

## Name mapping

Some models carry, in the result files, the technical name of their decomposition (in French):

| Name in the report | Name in `results/data/` |
|---|---|
| Sheared reservoir (part 5) | « Iwasawa : cisaillement (N) » (`form="iwasawa_N"`) |
| Sheared reservoir + amplitudes | « Iwasawa : amplitudes et cisaillement (A et N) » |
| Nilpotent reservoir (part 6) | « Jordan-Chevalley + partie nilpotente apprise » (`form="jordan_SN"`) |
| Tuned on the eigenvalues | « accordé, valeurs propres (Jordan-Chevalley, partie semi-simple) » |
| Reservoir with learned biases and inputs | « contrôle : biais + entrées W_in, sans g » (`learn_g=False`) |
| Random relay, learned relay, cross duos | « forme 3 », « forme 4 », « formes 1 et 2 » |

## Code layout

```
partially-frozen-rnn/
├── src/hfrnn/
│   ├── core.py            # reservoir, ridge, backpropagation through time, Adam
│   ├── tasks.py           # Mackey-Glass, NARMA10, memory, Fibonacci (demo), real series
│   ├── task_map.py        # the twelve tasks of the map of tasks and the linear regression baseline
│   ├── params.py          # the models of parts 1 to 3 (numpy) and training
│   ├── runner.py          # parallel experiment grids, cache, paired comparisons
│   ├── torch_models.py    # PyTorch version (GPU)
│   ├── torch_baselines.py # GRU, LSTM, LRU, and the linear tuned reservoir
│   ├── torch_tuned.py     # the tuned reservoir and its extensions (part 4)
│   ├── torch_decomp.py    # sheared and nilpotent reservoirs, full or banded (parts 5 and 6)
│   ├── torch_shear.py     # gated sheared reservoir: blocks of a GRU frozen, sheared or learned (part 5)
│   ├── torch_subspace.py  # learning in a given subspace, block unfreezing (part 9)
│   ├── structure_signals.py  # tangent spaces of the structures, gradients at the start, signals (part 9)
│   └── torch_coupled.py   # the four couplings and the squared readout (part 7)
├── experiments/           # one script per experiment
├── notebooks/             # the Colab notebook
├── tests/
├── rapport/               # the report, in French
├── report/                # the report, in English
├── results/               # figures and data
└── fausses_pistes/        # abandoned attempts, and why
```

The [`fausses_pistes/`](../fausses_pistes/README.en.md) folder keeps the attempts that led to misleading conclusions.
