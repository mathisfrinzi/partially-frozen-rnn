[Contents](../README.en.md#the-report) · [← Part 7](7_couplage.md) · [Part 9 →](9_structure.md) · [Français](../rapport/8_synthese.md)

# Part 8 — Synthesis: all the families, beyond synthetic data, and against existing models

![Learning little, from randomness](../results/figures/pitch.png)

> **What to remember**
>
> - On synthetic series, almost every way of staying between a reservoir and an RNN beats the RNN that learns everything. Each family has its tasks: the sculpted reservoir for memory, the tuned one for Mackey-Glass and NARMA10, the banded shear and the relays for Lorenz and Hénon; the banded shear even beats the tuned reservoir on all five tasks at 500 neurons, but this gain comes from a small sparse correction of the recurrence, not from the shear (part 5.5).
> - Against existing models, with the same search budget for all, the tuned reservoir (1,500 parameters) does better than the GRU on Mackey-Glass and than the LRU on NARMA10, and the banded shear (6,000 parameters) 3 times and 2 times better. The well-tuned LRU wins on memory, except against a linear variant with an input gate, and the GRU dominates chaotic systems built from products.
> - On non-synthetic data, the result is mostly negative: no gain in classification, no trained model beats the classic reservoir on three real series, better than the GRU on one physical system (Silverbox, with the banded shear, 127 times fewer parameters) and a clear defeat on another (Wiener-Hammerstein). The mechanisms at play are, for the most part, already known.

## 8.1 All the families side by side

At 100 neurons, each family against the RNN that learns everything, on the five synthetic tasks:

![Each family against the RNN](../results/figures/bilan_familles.png)

| Family (part) | What it learns in the recurrence | Where it wins | Where it loses |
|---|---|---|---|
| Braked RNN, mixture (1) | everything, slowly | Mackey-Glass, NARMA10, memory (2 to 4 times better than the RNN) | nothing clear; overtaken at 500 neurons |
| Sculpted reservoir (2) | an attenuation per connection | **memory, at every size**; NARMA10 at small size | costs N² parameters |
| Rotated reservoir (3) | a rotation | Hénon (4.4 times), Mackey-Glass | memory (10 times worse), MNIST |
| Tuned reservoir (4) | an amplitude per direction (and above all the biases and the inputs) | **Mackey-Glass and NARMA10 at 500 neurons** with 1,500 parameters | Lorenz (4.7 times worse than the RNN at 100 neurons) |
| Sheared reservoir (5) | a triangular shear, full or banded; in fact, a few thousand sparse recurrent coefficients | **banded, all five tasks at 500 neurons**, with a single k; Mackey-Glass and NARMA10 against the tuned GRU and LRU; Silverbox | full: overfits; Lorenz and Hénon against the GRU; Wiener-Hammerstein |
| Nilpotent reservoir (6) | eigenmodes + nilpotent part | Lorenz | Hénon; full: collapses at 500 neurons and on all real systems |
| Relays, squared readout (7) | a deeper transition, products | Hénon (93 times the RNN), Lorenz | memory; nothing transfers to real data |

The per-task syntheses gather all the experiments, each panel keeping its own protocol (size, duration, number of seeds):
models are compared within a panel, not from one panel to another.

![Synthesis, Mackey-Glass](../results/figures/synthesis_mackey_glass.png)

![Synthesis, NARMA10](../results/figures/synthesis_narma10.png)

![Synthesis, memory](../results/figures/synthesis_memoire.png)

## 8.2 Against existing models

**First protocol: four settings for each existing model, chosen on the test set** (500 neurons, 4,000 steps, 3 seeds). This
choice favours the existing models, which get four settings (two learning rates, two initializations, or one and two layers
for the LRU) against a single one for ours.

| N = 500, 4,000 steps | Recurrent parameters | Mackey-Glass | NARMA10 | Memory |
|---|---|---|---|---|
| RNN (best setting) | 251,000 | 7.8e-7 | 0.098 | 0.017 |
| GRU (best setting) | 754,500 | **6.4e-8** | 0.083 | 0.515 |
| LSTM (best setting) | 1,006,000 | 7.9e-8 | 0.127 | 0.210 |
| LRU (best setting) | 251,750 | 8.4e-5 | 0.023 | 0.0050 |
| Part 1: braked RNN | 251,000 | 3.8e-7 | 0.058 | 0.0014 |
| Part 1: rank-4 corrected reservoir | 5,000 | 9.4e-8 | 0.018 | 0.00054 |
| Part 2: fully sculpted reservoir | 251,000 | 1.0e-7 | 0.036 | **0.00018** |
| Part 4: tuned reservoir | **1,500** | 6.7e-8 | **0.0095** | 0.00053 |
| Part 4: selective tuned | 2,000 | 4.0e-8 | 0.0115 | 0.00078 |
| Part 4: gated tuned | 4,500 | 5.1e-8 | 0.0085 | 0.00051 |

![Comparison with existing models](../results/figures/comparisons_n500.png)

In this protocol, the tuned reservoir matches the GRU on Mackey-Glass, and all our models seem far ahead on NARMA10 and memory.
Two controls put this table in perspective: with optimal stopping (chosen on the test set for each model), the advantage of the
tuned reservoir on NARMA10 falls from 2.4 to 1.3 times, part of it coming from the fact that it does not overfit; and the fair
search below reverses the result on memory.

**Second protocol: the same search budget for all.** Each model gets its default configuration plus 12 random trials
(learning rate and initialization for the GRU; learning rate, layers and eigenvalue radii for the LRU; learning rate, learning
rates of the amplitudes and of the inputs, spectral radius, input scale and density for the tuned reservoir), judged on a
**validation** set taken at the end of the training part. The best configuration is retrained on **5 seeds** and measured once
on the test set. Two existing models additionally search their **number of units** (20 to 500). 500 neurons, 1,500 steps.
Paired gap to the best existing model (negative = better):

| Task | GRU | GRU, searched size | LRU | LRU, searched size | **Tuned (1,500 par.)** | Tuned with input gate (2,500) | Learned biases and inputs, without g (1,000) |
|---|---|---|---|---|---|---|---|
| Mackey-Glass | 1.4e-7 | 1.25e-7 (500 units, ref.) | 8.6e-6 | 5.4e-5 (100 units) | **7.0e-8 (−0.25 ± 0.06)** | **6.0e-8 (−0.32 ± 0.06)** | **6.4e-8 (−0.29 ± 0.09)** |
| NARMA10 | 0.093 | 0.068 (40 units) | 0.0137 (ref.) | 0.017 (40 units) | **0.0080 (−0.23 ± 0.03)** | **0.0081 (−0.23 ± 0.03)** | 0.0156 (+0.06 ± 0.01) |
| Memory | 0.54 | 0.0061 (20 units) | 2.3e-4 | **1.0e-4 (200 units, ref.)** | 6.5e-4 (**+0.80 ± 0.05**) | 4.8e-4 (+0.67 ± 0.04) | 7.5e-4 (+0.87 ± 0.06) |

![Fair search](../results/figures/recherche_equitable.png)

- **Mackey-Glass**: the tuned reservoir does 1.8 times better than the best GRU (the search keeps 500 units), without any
  tuning of its own (its default configuration is the best on validation).
- **NARMA10**: the tuned reservoir does 1.7 times better than the LRU, and 8 times better than the best GRU of searched size
  (40 units). A 60-unit GRU, never drawn by the search, reached 0.0063 in part 4.9: the GRU has a narrow and unstable size
  range in which it succeeds on this task.
- **Memory**: **the well-tuned LRU wins** (1.0e-4 at 200 units, 6 times better than the tuned reservoir, 4.7 times better than
  the input gate). The previous comparisons, where the LRU stayed at 0.005, disadvantaged it. The GRU remains bad, except one
  lucky trial at 20 units.
- **What the searched size changes**: on NARMA10, the GRU gains only 1.4 times over the 500-unit one; on memory, it improves
  90 times thanks to a trial at 20 units, but stays 9 times behind the tuned reservoir; the LRU of searched size gains 2.3 times
  on memory.
- **The reservoir with learned biases and inputs** (strictly random recurrence, 1,000 parameters) matches the tuned reservoir
  on Mackey-Glass and memory, and does 2 times worse on NARMA10, where it joins the LRU. On these three tasks, most of the
  advantage over existing models therefore does not require learning the recurrence.
- **The sheared models with the same budget** (k included in the search; details in part 5.9): the banded shear does 3.0 times
  better than the GRU on Mackey-Glass and 2.1 times better than the LRU on NARMA10; it loses 7 times against the LRU on memory.
  The **linear sheared reservoir with an input gate** does 8.3 times better than the LRU on NARMA10 and **23 times better on
  memory** (16,535 parameters against 161,000).
- **Lorenz and Hénon**, added to the search (8 trials for all): the tuned GRU wins, 16 times ahead of the banded shear on
  Lorenz; on Hénon, the sheared reservoir is 1.5 times behind the GRU (not significant), the tuned reservoir and the reservoir
  with learned biases and inputs 4 times.
- The fully sculpted reservoir did not receive this search budget.

## 8.3 Beyond synthetic tasks

![Beyond synthetic tasks](../results/figures/synthesis_beyond.png)

### Classification: no gain

- **Row-by-row MNIST** (28 steps, 256 neurons). With the same settings for all, the patched and sculpted reservoirs match the
  RNN (4.0 and 4.5 % error against 4.4 %), the tuned one is at 5.1 %, the rotated one fails (11.7 %). With the same search
  budget (24 trials on validation), the tuned reservoir stays at 3.1 %, the RNN goes down to 2.2 % and the GRU to 1.5 %;
  enlarging the tuned reservoir to 1,024 neurons does not help. The randomness of the directions limits what N amplitudes can
  do when **representations must be built**.
- **Pixel-by-pixel MNIST** (784 steps, 128 neurons, 3 epochs): the LRU wins (28.3 %), the linear tuned reservoir is 6 points
  behind with 52 times fewer parameters, the tuned reservoir beats the GRU and the RNN (46.7 % against 59.3 and 64.9 %).
  Nothing is close to the state of the art: it is a short-protocol ranking.
- **UEA archive** (ten multivariate datasets, 128 neurons): the GRU is the best (mean rank 2.2, first on 6 datasets); the tuned
  reservoir is on par with the RNN and the LRU (ranks 3.4, 3.3 and 3.1) with 10 times fewer parameters, and wins on the two
  shortest datasets.

![UEA archive](../results/figures/uea_n128.png)

### Real time series: the classic reservoir is the best

Oil temperature of a transformer (ETTh1, at 24 h), minimum temperature in Melbourne (at 1 day), sunspots (at 1 month); the
target is the variation, and an NMSE of 1 corresponds to the naive forecast "nothing changes". 100 neurons, 5 seeds.

| | ETTh1 | Melbourne | Sunspots |
|---|---|---|---|
| Classic reservoir | **1.01** | **0.77** | **1.08** |
| Best RNN | 1.18 | 0.97 | 1.11 |
| Braked RNN | 1.20 | 1.13 | 1.27 |
| Rank-4 corrected reservoir | 1.12 | 1.11 | 1.10 |
| Fully sculpted reservoir | 1.15 | 0.88 | 1.12 |
| Tuned reservoir | 1.16 | 0.83 | 1.09 |

No model beats the naive forecast on ETTh1 and sunspots; on Melbourne, the best is the classic reservoir, without any
recurrent learning. The trained models overfit these short, noisy series; the most constrained ones fare best. The protocol
(no early stopping, a single variable, short horizons) is not suited as it is.

![Real series](../results/figures/real_data.png)

### Map of tasks: for which data?

Twelve synthetic tasks from different families (chaos, identification, telecommunications, noise, little data, long memory),
to see on which kind of data the tuned reservoir keeps an advantage. Baseline to beat: a **linear regression** on a window of
past values. 500 neurons, 3 seeds.

| Task | Linear | Classic reservoir | RNN | GRU | LRU | Tuned | Linear tuned | Tuned against the best existing |
|---|---|---|---|---|---|---|---|---|
| Lorenz (+10) | 0.12 | 5.3e-4 | 4.6e-5 | **2.2e-7** | 0.012 | 4.2e-5 | 4.6e-3 | loses (×190) |
| Hénon (+3) | 0.80 | 0.52 | 0.86 | **8.4e-4** | 0.024 | 0.064 | 0.073 | loses (×77) |
| Lorenz-96, 40 inputs | **0.24** | 0.66 | 0.64 | 0.45 | 0.52 | 0.55 | 0.55 | loses (×1.2) |
| Wiener-Hammerstein (simulated) | 0.0081 | 0.058 | 0.015 | **5.9e-4** | 0.0083 | 0.0013 | 0.0056 | loses (×2.3) |
| Channel equalization | 0.012 | 0.013 | 0.010 | 0.0049 | 0.011 | **0.0036** | 0.0087 | wins (×1.35) |
| Wireless channel (CSI) | **0.0083** | 0.043 | 0.045 | 0.15 | 0.015 | 0.040 | 0.020 | loses (×2.7) |
| Power amplifier | 0.47 | 0.20 | 0.26 | 0.26 | 0.68 | **0.0035** | 0.75 | wins (×74) |
| Mackey-Glass, 20 % noisy input | 0.25 | 0.032 | **0.015** | 0.016 | 0.033 | 0.022 | 0.030 | loses (×1.4) |
| Mackey-Glass, 300 steps | 0.36 | 8.0e-4 | 1.5e-4 | 4.0e-5 | 0.0070 | **2.9e-5** | 0.013 | tie |
| NARMA10, 300 steps | 0.21 | 0.53 | 0.32 | 0.46 | 0.128 | 0.233 | **0.111** | loses (×1.8) |
| Memory, delay 30 | **1.2e-19** | 1.18 | 1.32 | 1.37 | 0.249 | 0.179 | 0.117 | wins (×1.4) |
| Memory, delay 60 | **1.3e-19** | 1.24 | 1.66 | 1.51 | 1.37 | 1.98 | 0.88 | loses (×1.4) |

![Map of tasks](../results/figures/task_map_n500.png)

- **The tuned reservoir wins on only three tasks out of twelve** (channel equalization, amplifier, memory at 30 steps) and ties
  on a fourth (Mackey-Glass with 300 steps). On Lorenz and Hénon, made of **products of variables**, which a multiplicative gate
  represents directly, the GRU crushes the others.
- Pure memories are solved exactly by linear regression: no recurrent model is of interest there.
- The gates added to the tuned reservoir only matter on Hénon (1.8 times) and Wiener-Hammerstein (1.2 times): they are not what
  the tuned reservoir lacks on chaotic systems (the GRU stays 120 and 40 times ahead).

**The amplifier, more closely.** The 74-fold gap against the GRU resists longer training and two learning rates. But at 3,000
points, the GRU has a test error 638 times its training error (3.8 times for the tuned reservoir): it is mostly
**overfitting**. With 20,000 points, the gap falls to 6 times; with enriched inputs (I, Q, |x|, |x|², as in the models of the
field), the GRU almost joins the tuned reservoir (1.8 times). The generalized memory polynomial, the reference of the field, is
very bad with our (non-specialist) settings. The tuned reservoir is therefore not "exceptional" on this system: it learns well
with little data.

![Amplifier](../results/figures/amplifier_deep_dive.png)

### Real physical systems: a tie, a defeat

Three identification benchmarks ([nonlinearbenchmark.org](https://www.nonlinearbenchmark.org)), measured on real systems, in
**simulation** (the model only sees the input): Silverbox (an electronic Duffing oscillator, three tests), Wiener-Hammerstein
(filter-diode-filter), Cascaded Tanks (two water tanks, 1,024 training points). 500 neurons, 3 seeds, 1,500 steps; RMSE in the
unit of the system.

| Test RMSE | Silverbox, multisine (mV) | Silverbox, arrow with extrapolation (mV) | Silverbox, arrow without extrapolation (mV) | Wiener-Hammerstein (mV) | Cascaded Tanks (V) |
|---|---|---|---|---|---|
| Linear regression (window) | 7.1 | 14.4 | 6.7 | 43.4 | **0.65** |
| Classic reservoir | 29.0 | 33.7 | 26.3 | 83.4 | 2.50 |
| RNN, lr × 0.1 | 6.6 | 12.5 | 6.7 | 15.6 | diverges |
| GRU (754,500 parameters) | 4.9 | **7.6** | 4.1 | **3.2** | diverges (1 seed out of 3) |
| LRU | 7.8 | 14.8 | 7.1 | 22.3 | 2.00 |
| Tuned reservoir (1,500) | 4.3 | 9.9 | 4.0 | 11.8 | 1.69 |
| Gated tuned | 4.1 | 9.9 | 4.0 | 10.4 | 1.70 |
| Tuned + squared readout (part 7) | 4.2 | 10.1 | 4.0 | 11.2 | 2.06 |
| Relay + tuned + squared readout (part 7) | 6.4 | 15.2 | 6.3 | 21.2 | 2.46 |
| Nilpotent reservoir (part 6) | 9.7 | 15.4 | 9.8 | 19.1 | 658 |
| Learned biases and inputs, without g (part 4, 1,000) | 13.3 | 17.2 | 11.2 | 22.6 | 3.03 |
| Banded shear, k = 3 (part 5, 2,494) | 3.6 | 9.1 | 3.6 | 9.8 | 3.54 |
| **Banded shear, k = 10 (part 5, 5,945)** | **2.9** | 8.1 | **2.8** | 7.8 | 4.27 |
| Full shear (part 5, 125,750) | 10.8 | 15.4 | 10.4 | 18.3 | 440 |
| *Published: best linear model* | | *13.5* | | *56.2* | *0.75* |
| *Published: GRU (Champneys et al., 2024)* | *1.49* | *2.80* | *0.95* | *3.97* | *0.40* |
| *Published: state of the art* | | *0.26* | *0.32* | *0.24* | *0.18 to 0.34* |

![Real systems](../results/figures/nonlinear_benchmarks.png)

- **Silverbox: the tuned reservoir matches the GRU**, with 500 times fewer parameters (paired gaps −0.06 ± 0.08 and
  0.00 ± 0.07 on two tests), and the **banded shear, k = 10, does better than it** on two tests out of three (2.9 against
  4.9 mV on the multisine), with 127 times fewer parameters. The GRU wins when extrapolation outside the amplitude range seen in
  training is needed (7.6 against 8.1 mV).
- **Wiener-Hammerstein: the GRU clearly wins**, 3.2 to 3.7 times, and even does a little better than the published GRU.
- **Cascaded Tanks: no recurrent model beats linear regression.**
- Our linear regression recovers the best published linear model: the protocol is consistent. But all our models stay 7 to 40
  times behind the state of the art, and the published GRU, trained for 10,000 to 20,000 epochs with a size chosen by
  cross-validation, does 3 to 4 times better than ours on Silverbox.
- **Only the banded shear transfers a synthetic gain** (1.5 times the tuned reservoir on Silverbox and Wiener-Hammerstein, with
  k = 10). The others do not transfer: squared readout, no effect; relays, 1.5 to 1.8 times worse; full nilpotent and full
  shear, 1.5 to 2.6 times worse.
- **The amplitudes g matter here for the tuned reservoir**: without them, the error is 1.7 to 3 times larger, whereas on
  well-tuned synthetic series they only served on NARMA10. Added to the sheared reservoir, they bring almost nothing.
- **A fourth system, EMPS**, which had not been used for any tuning (a positioning table in closed loop): all models fail, GRU
  included (NMSE above 1); no conclusion.

## 8.4 Position relative to existing work

**Three families close to this work.**

- **Intrinsic plasticity** ([Wardermann and Steil, ESANN 2007](https://www.esann.org/sites/default/files/proceedings/legacy/es2007-98.pdf);
  Schrauwen et al., 2008). A local, unsupervised rule adapts the **gain and the bias** of each neuron of a reservoir, by
  gradient descent on a Kullback-Leibler divergence to a target output distribution; it improves NARMA10. Part 4.2 shows that a
  large part of the gain of the tuned reservoir comes from learning the biases and the inputs: adapting one bias per neuron is
  therefore known, with another criterion. An IP baseline next to the tuned reservoir remains to be measured.
- **Slow gradient descent on the weights of a reservoir**: Atiya-Parlos recurrent learning and backpropagation-decorrelation
  (BPDC, Steil), which learn the output quickly and the internal weights slowly
  ([Lukoševičius and Jaeger, 2009](https://www.ai.rug.nl/minds/uploads/2261_LukoseviciusJaeger09.pdf)): the braked RNN of part
  1 belongs to this line.
- **LottaLoRA** (*A Little Rank Goes a Long Way: Random Scaffolds with LoRA Adapters Are All You Need*,
  [arXiv:2604.08749](https://arxiv.org/abs/2604.08749)). Each weight is `β·W_seed + (α/r)·B·A`, with `W_seed` random and frozen,
  `B·A` of rank r learned and one scalar β per layer; 96 to 100 % of full training with 0.5 to 40 % of the parameters on nine
  benchmarks, and a scaffold that must remain **static**. A single recurrent experiment: a frozen random CfC network with rank-r
  adapters, in **classification** (ICU mortality, PhysioNet 2012); AUROC 0.832 at rank 1 against 0.836 for the trained network.
  No classic echo state reservoir, no prediction of chaotic series, no equal-budget comparison with a GRU, no singular
  amplitudes. The principle "frozen random base + rank-r correction" is therefore not to be claimed: the rank-r corrected
  reservoir (part 1) and the rank-r tuned reservoir (part 4.9) are instances of it.

**Family by family.** The details are in the assessment of each part; in summary:

| Part | Closest neighbour | What remains specific to this work |
|---|---|---|
| 1 · Mixing | μP (Yang et al.); Atiya-Parlos, BPDC | the practical equivalence mixing = slowing down with Adam; the tuned mixture, which finds the rule by itself |
| 2 · Sculpted | supermasks, Piggyback | a **continuous, bounded** mask on a reservoir, compared with the binary mask and the free gain |
| 3 · Rotated | orthogonal RNNs (Lezcano-Casado) | the dissociation "spectrum for memory, orientation for prediction" on the same reservoir |
| 4 · Tuned | VeRA, SVFit, SVFT; intrinsic plasticity; LottaLoRA | the transfer to the recurrence of a reservoir; the "who learns what" controls; the linear tuned reservoir against the LRU |
| 5 · Sheared | nnRNN (Kerg et al.); non-normal chains (Ganguli et al.); sparse fine-tuning; gated linear RNNs (minGRU, LRU) | the Iwasawa N factor of a reservoir, with no precedent found, but the controls show that it does not explain the gain; a sparse correction of a few thousand coefficients, measured against the tuned GRU and LRU |
| 6 · Nilpotent | nnRNN | nothing new: a variant applied to a reservoir |
| 7 · Coupling | DT-RNN, NVAR, state-feedback ESN | a frozen random layer inside the learned transition; the comparison of the four couplings |

The building blocks are known separately; what might be new is narrow: their application to a random reservoir, the controls
that show where the gain comes from, and the equal-protocol comparisons. The literature search is not exhaustive (several
papers were only read through their abstract; see the [references](references.md)).

## 8.5 What holds, and the limits

**What holds** (several seeds, clear paired gaps, several sizes):
- At equal state size, the models that stay close to a random reservoir largely beat the RNN that learns everything, at almost
  every size from 100 to 2,000 neurons, on the synthetic tasks.
- On synthetic series, the gain of the tuned reservoir comes mostly from the biases and the inputs; well tuned, a reservoir in
  which only the biases and the inputs learn matches it except on NARMA10. On real systems, the amplitudes become
  indispensable.
- A sparse correction of a few thousand recurrent coefficients (banded shear, or the same coefficients learned at random in
  `W₀`) beats the tuned reservoir on all five synthetic tasks at 500 neurons, with a single setting for all, on regenerated
  series, and does better than the GRU on Silverbox. With the same search budget, it beats the GRU on Mackey-Glass (3 times)
  and the LRU on NARMA10 (2 times).
- With the same search budget, 1,500 parameters do better than the GRU on Mackey-Glass and than the LRU on NARMA10.
- Frozen random directions and learned eigenvalues: the principle matches the full LRU with 200 times fewer parameters.
- A shared reservoir and one stored setting per task beat one RNN per task, without forgetting.

**What does not hold, or not yet**:
- Memory against a well-tuned LRU; chaotic systems built from products against a large GRU (at equal budget, the rank-r
  correction without g beats it by 16 to 17 times up to 33,000 parameters); classification; noisy real series;
  Wiener-Hammerstein, where the GRU stays 3 times ahead of all our models.
- The gains of full non-normality (parts 5 and 6), of the squared readout and of the relays (part 7) remain synthetic.
- The K·A·N structure: the gain of the sheared reservoir comes neither from the band nor from the basis of the decomposition
  (part 5.5).

**General limits of the protocol.**
- A single optimizer (Adam) and fixed durations (1,500 steps; 4,000 for the first comparison with existing models; 3 epochs on
  MNIST, 10 with the search).
- Tasks with 3,000 training points, which limit what a large network can get out of them.
- 3 seeds in most PyTorch experiments (5 in the fair search, 8 at 100 neurons in numpy).
- The amplifier and the channel are simulated; the published references for the real systems are specialized models, tuned at
  length.
- No proven theory: the explanations come from controls and simple properties (Perron-Frobenius bound, rotation invariance,
  transient amplification), not from theorems on performance.

---

[Contents](../README.en.md#the-report) · [← Part 7](7_couplage.md) · [Part 9 →](9_structure.md)
