# partially-frozen-rnn

*Between a reservoir and an RNN: start from a frozen random recurrence, and learn only one chosen piece.*

[Version française](README.md) · **English**

[![tests](https://github.com/mathisfrinzi/partially-frozen-rnn/actions/workflows/tests.yml/badge.svg)](.github/workflows/tests.yml)

A **reservoir** (Echo State Network) draws its recurrent matrix at random and learns only its readout. An **RNN** learns
everything. This repository studies what lies in between. Each part fixes what the recurrence is allowed to learn (a speed,
an attenuation per connection, a rotation, a gain per direction, a shear, a nilpotent part), measures what it is worth with
paired seeds, says on which tasks it works, and relates the result to theory and to the literature.

![Between a reservoir and an RNN](results/figures/schema_continuum.png)

## In short

> **In one sentence: ease of learning beats the geometry of the target, even when the exact solution lies in the family.**
> Where the banded sheared reservoir wins, the same number of coefficients learned at random positions does as well
> (part 5.5); no choice of directions, even one dictated by the gradient, beats chance at equal dimension (part 9.5); and when
> a structure is planted in the task, the student that contains the exact solution wins clearly in no case: bands of a few
> hundred coefficients win instead (part 9.7, at 100 neurons).

- **Staying in between beats the RNN that learns everything** on synthetic series, at almost every size from 100 to 2,000
  neurons, but each family has its own tasks.

  ![Each family against the RNN](results/figures/bilan_familles.png)

- **Mixing is slowing down** (part 1): with Adam, `(1−α)·W₀ + α·ΔW` amounts to learning α times more slowly. The rule that
  follows, free inputs and a recurrence 10 times slower, is very close to the μP prescription.
- **The sculpted reservoir** (part 2), which only learns an attenuation between 0 and 1 per connection, is the best model of
  the study on memory, at every size; the bound is what matters.
- **The rotated reservoir** (part 3) keeps the spectrum and only learns the orientation: useful for chaotic prediction,
  useless for memory.
- **The tuned reservoir** (part 4) only learns one gain per singular direction: with 1,500 parameters at 500 neurons, it
  beats the GRU on Mackey-Glass (1.8 times) and the LRU on NARMA10 (1.7 times) with the same search budget. **But on these
  series, the gain comes mostly from the biases and the inputs**: well tuned, a reservoir in which only the biases and `W_in`
  learn matches the tuned reservoir except on NARMA10, and the gains hurt on Lorenz and Hénon. On real systems, on the
  contrary, the gains are indispensable. The well-tuned LRU wins on memory.
- **The sheared and nilpotent reservoirs** (parts 5 and 6) learn one factor of the Iwasawa or Jordan-Chevalley
  decomposition. Full, they overfit. **Banded** (a few thousand parameters), the sheared reservoir beats the tuned one on the
  five synthetic tasks with a single setting, holds on regenerated series, beats the GRU on Silverbox and, with the same
  search budget, the GRU on Mackey-Glass and the LRU on NARMA10. **But controls show that it is not the shear**: the same
  number of coefficients learned at random positions, even directly in `W₀`, does as well. The GRU keeps Lorenz and Hénon.
- **Coupling a frozen reservoir and an RNN** (part 7): a random relay inside the transition beats the GRU on Hénon, but a fully
  learned relay does as well; depth is what matters.
- **Beyond synthetic data** (part 8): better than the GRU on a real physical system (Silverbox) with 127 times fewer
  parameters, a clear defeat on another one (Wiener-Hammerstein), no gain in classification. The mechanisms at play are, for
  the most part, already known (μP, supermasks, singular value adaptation, intrinsic plasticity, LottaLoRA, nnRNN); what
  remains specific to this work is narrow.
- **Can the task choose its own structure?** (part 9): no, not from the gradient. Projected onto the tangent space of each
  structure, it does not say which one to learn better than their size does, neither at the start nor during training; nor
  does it point to a subspace, including through a Lie decomposition learned on the task (PCA of the tangent `M⁻¹G`). Even on
  tasks where a structure is planted and reachable, neither the gradient nor training recovers it: at equal budget, bands
  win. Two promising leads were checked on fresh seeds, and fell. Ease of learning beats the geometry of the target, even
  when the exact solution lies in the family.

## The report

| | Part | What the recurrence learns | Diagram |
|---|---|---|---|
| 0 | [Introduction](report/0_introduction.md) | the question, the notation, the protocol | [principle](results/figures/schema_principe.png) |
| 1 | [Mixing is slowing down](report/1_melanger.md) | everything, slowly; patched, rank-r corrected | [diagram](results/figures/schema_partie1_freine.png) |
| 2 | [The sculpted reservoir](report/2_sculpte.md) | an attenuation between 0 and 1 per connection | [diagram](results/figures/schema_sculpte.png) |
| 3 | [The rotated reservoir](report/3_tourne.md) | a rotation, frozen spectrum | [diagram](results/figures/schema_tourne.png) |
| 4 | [The tuned reservoir](report/4_accorde.md) | one gain per direction; who learns what; gates, linear, size, several tasks, equal budget | [diagram](results/figures/schema_accorde.png) |
| 5 | [The sheared reservoir](report/5_cisaille.md) | a triangular shear (Iwasawa) | [diagram](results/figures/schema_cisaille.png) |
| 6 | [The nilpotent reservoir](report/6_nilpotent.md) | eigenmodes and nilpotent part (Jordan-Chevalley) | [diagram](results/figures/schema_nilpotent.png) |
| 7 | [Coupling a reservoir and an RNN](report/7_couplage.md) | two separate networks: duos, relays, squared readout | [diagram](results/figures/schema_couplage.png) |
| 8 | [Synthesis](report/8_synthese.md) | all families, existing models, non-synthetic data, position with respect to prior work | [figure](results/figures/recherche_equitable.png) |
| 9 | [Can the task choose its own structure?](report/9_structure.md) | does the gradient say which structure or subspace to learn? even when the structure is planted in the task? | [figure](results/figures/enseignants.png) |
| | [Appendix: pending measurements and leads](report/annexe_pistes.md) · [References](report/references.md) · [Reproduce](report/reproduire.md) · [Dead ends](fausses_pistes/README.en.md) | | |

Each part about a reservoir starts with its diagram and with "what to remember", and ends with the same assessment: where it
works, the limits, the link with theory.

The figures are those of the French report: their labels are in French.

![How much each model learns](results/figures/schema_budget.png)

## Reproduce

```bash
pip install -e ".[dev]"     # numpy, scipy, matplotlib, pytest; ".[dev,gpu]" for PyTorch
pytest                      # a few minutes
python experiments/38_schemas.py            # all the explanatory diagrams, in a few seconds
python experiments/42_figures_rapport.py    # the summary figures, from the saved results
```

Computations at 500 neurons and more run on GPU; the GPU blocks are in
[`notebooks/colab_gpu.ipynb`](notebooks/colab_gpu.ipynb). The list of scripts, part by part, is in
[report/reproduire.md](report/reproduire.md).

## Repository layout

```
partially-frozen-rnn/
├── README.md            # French version of this page
├── README.en.md         # this page
├── rapport/             # the report in French, one file per part
├── report/              # the report in English, same files
├── src/hfrnn/           # the code: reservoirs, models (numpy and PyTorch), tasks
├── experiments/         # one script per experiment, cited in the report
├── notebooks/           # the Colab notebook for the GPU measurements
├── results/             # figures (including the diagrams) and data
├── tests/               # checks (gradients, numpy = PyTorch, each model)
└── fausses_pistes/      # abandoned attempts, and why
```
