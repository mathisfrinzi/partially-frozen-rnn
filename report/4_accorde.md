[Contents](../README.en.md#the-report) · [← Part 3](3_tourne.md) · [Part 5 →](5_cisaille.md) · [Français](../rapport/4_accorde.md)

# Part 4 — The tuned reservoir: learning only one gain per direction

![Tuned reservoir](../results/figures/schema_accorde.png)

> **What to remember**
>
> - The random singular directions of the reservoir are kept (`W₀ = U·diag(σ)·Vᵀ`) and only the gain of each one is learned: `W = U·diag(σ ⊙ g)·Vᵀ`, together with `W_in` and the biases. At 500 neurons, 1,500 learned parameters are enough to beat the RNN by far, and to do better than the GRU and the LRU on Mackey-Glass and NARMA10 in a fair search.
> - **On synthetic series, the gain comes mostly from the biases and the inputs, not from the gains.** The gains g alone are barely worth the classic reservoir; a reservoir in which only the biases and `W_in` learn does most of the work and, with the same search budget, matches the tuned reservoir on Mackey-Glass and memory. g is clearly useful only on NARMA10 (2 times), and **hurts** on Lorenz and Hénon.
> - **On real physical systems, it is the opposite**: without g, the error is 1.7 to 3 times larger on Silverbox and Wiener-Hammerstein. There, the gains make up most of the advantage of the tuned reservoir.
> - The extensions (gates, linear recurrence, per-task settings, rank-r correction) give targeted gains; the clearest is the linear tuned reservoir, which matches the full LRU with 200 times fewer parameters.

A matrix rotates, stretches, then rotates again: this is its singular value decomposition. The tuned reservoir keeps the two
random rotations and learns only the stretch of each direction. It is the "gains" half of the question of part 3, which on
the contrary froze the gains to learn only the directions.

```
x(t+1) = tanh( W_in·u(t) + U·diag(σ ⊙ g)·Vᵀ·x(t) + b )        learned: g (N values), W_in, b
```

## 4.1 The model, and what it is worth

**At 100 neurons**, it only matches the RNN (table 3.1: gaps 0.00, −0.11, +0.02). **At 500 neurons**, it becomes the best
model of the study on Mackey-Glass and one of the two best on NARMA10 (3 seeds, against the best of four learning rates of
the RNN):

| N = 500 | Mackey-Glass | × less error than the RNN | NARMA10 | × less error than the RNN |
|---|---|---|---|---|
| Classic reservoir | 1.3e-5 | 0.1 | 0.241 | 0.4 |
| Best RNN | 1.1e-6 | reference | 0.0945 | reference |
| Part 1: braked RNN | 1.9e-7 | 6.1 | 0.0315 | 3.0 |
| Part 1: rank-4 corrected reservoir | 8.4e-8 | 13 | **0.0101** | **9.3** |
| Part 2: fully sculpted reservoir | 8.7e-8 | 13 | 0.0173 | 5.3 |
| Part 3: rotated reservoir | 1.7e-7 | 6.6 | 0.0383 | 2.4 |
| **Tuned reservoir** | **7.5e-8** | **15** | 0.0102 | 9.1 |

It learns only 500 gains on the directions of `W₀` (plus `W_in` and `b`, 1,500 parameters in all), against 250,000 recurrent
weights for the RNN. Under an identical protocol (500 neurons, 1,500 steps), its variants alone form the error/parameter
front on the three main tasks; the only exception is the fully sculpted reservoir on memory.

![Error against parameters](../results/figures/synthesis_pareto.png)

Against GRUs, LSTMs and LRUs, with the same search budget for all (part 8.2), it does 1.8 times better than the best existing
model on Mackey-Glass and 1.7 times on NARMA10, but the well-tuned LRU beats it on memory (6 times).

## 4.2 Where does the gain come from?

This is the central question of this part, and the answer corrects the naive reading of the model.

**Who learns what.** In the tuned reservoir, three things learn: the gains `g`, the biases `b` and the inputs `W_in`. They
are switched on one by one, then in pairs, at 200 neurons (3 seeds). The three "without g" rows keep exactly the recurrence
of the reservoir (`W = W₀`).

| What is learned (parameters) | Mackey-Glass | NARMA10 | Memory | Lorenz | Hénon |
|---|---|---|---|---|---|
| *nothing: classic reservoir (0)* | 1.6e-4 | 0.26 | 0.52 | 1.5e-3 | 0.59 |
| g only (200) | 9.6e-5 | 0.22 | 0.105 | 9.3e-4 | 0.77 |
| biases only, without g (200) | 2.2e-5 | 0.167 | 0.518 | 4.2e-4 | 0.535 |
| W_in only, without g (200) | 7.4e-5 | 0.178 | 0.0061 | 1.7e-4 | 0.056 |
| **biases + W_in, without g (400)** | 5.4e-6 | 0.048 | 0.0068 | **2.7e-5** | **0.027** |
| g + biases (400) | 6.7e-6 | 0.038 | 0.110 | 2.6e-4 | 0.79 |
| g + W_in (400) | 4.1e-5 | 0.21 | **0.0013** | 3.2e-3 | 0.086 |
| **g + biases + W_in: the tuned reservoir (600)** | **3.6e-6** | **0.0156** | 0.0018 | 2.4e-4 | 0.057 |

![Who learns what](../results/figures/accorde_qui_apprend_quoi.png)

- **The gains g alone are worth almost nothing.** They do no better than the classic reservoir on NARMA10 (0.22 against 0.26)
  and do **worse** on Hénon (0.77 against 0.59); the gain is 1.6 times on Mackey-Glass and Lorenz, 5 times on memory.
- **Biases and inputs, without g, do most of the work.** With a strictly random recurrence, learning the biases and `W_in`
  divides the error by 30 on Mackey-Glass, 5 on NARMA10, 75 on memory, 55 on Lorenz and 22 on Hénon.
- **Which of the two matters depends on the task**: the bias on Mackey-Glass, NARMA10 and Lorenz; the inputs on memory
  (80 times) and Hénon (9 times).
- **What g adds changes sign with the task.** Relative to "biases + W_in", g helps on NARMA10 (3 times), memory (3.7 times)
  and a little on Mackey-Glass (1.5 times), but **hurts** on Lorenz (9 times) and Hénon (2 times): on these two tasks, the
  model without g, with 400 parameters, does better than the full tuned reservoir at 500 neurons.

**With the same search budget.** The table above keeps the default settings. At 500 neurons, with the fair search of
part 8.2 (12 trials on validation, 5 seeds), the reservoir with learned biases and inputs (1,000 parameters) does **as well as
the tuned reservoir on Mackey-Glass** (6.4e-8 against 7.0e-8, gap −0.04 ± 0.04) **and on memory** (7.5e-4 against 6.5e-4,
+0.06 ± 0.03), but **2 times worse on NARMA10** (0.0156 against 0.0080, +0.29 ± 0.04). Once each model is well tuned, the
specific contribution of the gains therefore shrinks to NARMA10.

**On real physical systems, however, g matters.** On the three systems of part 8.3 (500 neurons, 3 seeds), the reservoir with
learned biases and inputs does clearly worse than the tuned reservoir:

| Test RMSE | Silverbox, multisine | Silverbox, arrow with extrapolation | Silverbox, arrow without extrapolation | Wiener-Hammerstein | Cascaded Tanks |
|---|---|---|---|---|---|
| Tuned reservoir (1,500 par.) | 4.25 mV | 9.93 mV | 4.03 mV | 11.8 mV | 1.69 V |
| Learned biases and inputs, without g (1,000 par.) | 13.3 mV (×3.1) | 17.2 mV (×1.7) | 11.2 mV (×2.8) | 22.6 mV (×1.9) | 3.03 V (×1.8) |

On these data, it is the gains that allow the tuned reservoir to match the GRU on Silverbox; a reservoir in which only the
biases and the inputs learn is not enough there.

**The same observation, seen through the truncated tuned reservoir.** Only r directions out of 500 are tuned (r = 2, 10, 50),
chosen among the largest singular values, the smallest, or at random; `W_in`, the biases and the readout always learn. With
r = 2 smallest directions (σ ≈ 0, hence an almost unchanged recurrence), one already recovers **a third of the gain of the
full tuned reservoir on Mackey-Glass, 40 % on NARMA10 and 100 % on memory**. Which directions are tuned hardly matters: at
r = 2 and 10, the largest are worth chance, and the smallest are only 1.2 times worse.

| Tuned directions (500 neurons, 3 seeds) | Mackey-Glass | NARMA10 | Memory |
|---|---|---|---|
| *Classic reservoir* | 1.3e-5 | 0.241 | 0.221 |
| *Best RNN (251,000 parameters)* | 1.1e-6 | 0.0945 | 0.0363 |
| **Full tuned reservoir (500 directions)** | **7.5e-8** | **0.0102** | **0.00186** |
| r = 2: largest / smallest / random | 1.9e-7 / 2.3e-7 / 2.1e-7 | 0.022 / 0.027 / 0.026 | 0.0019 / 0.0019 / 0.0019 |
| r = 10: largest / smallest / random | 1.9e-7 / 2.3e-7 / 2.0e-7 | 0.020 / 0.027 / 0.024 | 0.0018 / 0.0019 / 0.0017 |
| r = 50: largest / smallest / random | 1.1e-7 / 2.3e-7 / 1.9e-7 | 0.016 / 0.026 / 0.017 | 0.0021 / 0.0019 / 0.0016 |

The smallest directions are a little worse than the largest (1.2 times at r ≤ 10, 1.6 to 2.1 times at r = 50, nothing on
memory); the largest do not beat chance at r = 2 and 10. At r = 10, the largest give back the spectrum controls below
(×0.40, ×0.50, ×1.03 against ×0.41, ×0.50, ×1.04): the two experiments agree.

**The spectrum controls** (500 neurons, 3 seeds; error factor relative to the tuned reservoir, above 1 = better):

| | Mackey-Glass | NARMA10 | Memory |
|---|---|---|---|
| Fourier directions instead of random ones | ×0.40 | ×0.95 | ×0.96 |
| *Fourier reservoir, untrained* | *×0.001* | *×0.05* | *×0.006* |
| Flat starting spectrum (all σ equal) | ×0.95 | ×1.00 | **×1.21** |
| Spread starting spectrum | ×0.39 | ×0.78 | ×0.34 |
| 100 / 50 / 10 tuned directions out of 500 | ×0.84 / ×0.70 / ×0.41 | ×0.71 / ×0.64 / ×0.50 | ×0.82 / ×0.87 / ×1.04 |
| Best RNN | ×0.07 | ×0.11 | ×0.05 |

![Spectrum controls](../results/figures/spectrum_controls_n500.png)

Random directions are not indispensable (Fourier directions do as well on NARMA10 and memory), and the starting spectrum does
not help: a neutral start (scaled orthogonal reservoir) is even the best on memory. Read together with the "who learns what"
table, these controls say the same thing: what remains when almost all the tuning of the recurrence is removed is a reservoir
whose biases and inputs learn.

## 4.3 What training does to the dynamics: memory capacity

A random signal u(t) is fed into the trained network and a linear readout is asked to recover u(t − k): the capacity at delay
k is C_k = 1 − NMSE, and their sum MC = Σ C_k is at most N (Jaeger, 2002). A degree-2 nonlinear capacity is also measured
(Dambre et al., 2012): recovering squares and products of past inputs. 100 neurons, 3 seeds, each model trained on each of
the three tasks.

| 100 neurons | trained on Mackey-Glass: memory / nonlinear | on NARMA10 | on memory |
|---|---|---|---|
| Classic reservoir (untrained) | 12.1 / 0.0 | 12.1 / 0.0 | 12.1 / 0.0 |
| RNN | 5.9 / 4.3 | 12.9 / 26.7 | 22.5 / 6.4 |
| Part 1: braked RNN | 9.0 / 21.0 | 14.0 / 45.1 | 26.4 / 8.3 |
| Part 1: rank-4 corrected reservoir | 8.3 / 18.0 | 14.8 / 43.1 | 28.3 / 5.3 |
| Part 2: fully sculpted reservoir | 7.8 / 18.0 | 13.7 / **48.3** | 24.4 / 14.5 |
| Part 3: rotated reservoir | 11.2 / 21.0 | 13.8 / 21.6 | 19.5 / 0.6 |
| **Tuned reservoir** | 9.1 / 20.6 | 14.9 / 41.4 | **31.1** / 4.7 |

![Memory capacity](../results/figures/memory_capacity.png)

Training shapes the dynamics according to the task: trained on memory, each model lengthens its memory; on NARMA10, it keeps
the same memory but gains a lot of nonlinear capacity; on Mackey-Glass, it sacrifices memory for computation. The tuned
reservoir gets the longest memory of all (31, against 22 for the RNN that learns its 10,000 weights), the rotated one the
shortest (19.5).

## 4.4 Thirteen variants, and combinations

![Variants of the tuned reservoir](../results/figures/tuned_gallery.png)

At 500 neurons (3 seeds), NMSE and paired gap to the tuned reservoir (negative = better; in bold: better beyond the noise):

| Variant | What changes | Parameters | Mackey-Glass | NARMA10 | Memory |
|---|---|---|---|---|---|
| **Tuned reservoir** | — | 1,500 | 7.4e-8 | 0.010 | 0.0019 |
| Bounded tuned | `g ∈ [0, 2]` | 1,500 | +0.15 | +0.04 | −0.02 |
| Tuned + leak | one learned speed per neuron | 2,000 | −0.04 | −0.01 | −0.03 |
| Tuned, eigenvalues | modulus and phase of each eigenmode | 2,000 | −0.03 | +0.31 | +0.35 |
| Parametric tuned | `g` = smooth function of 8 coefficients | 1,008 | +0.04 | +0.24 | +0.18 |
| Tuned + rank 4 | `+ A·Bᵀ` | 5,500 | +0.09 | +0.05 | **−0.08** |
| Deep tuned | two stacked tuned reservoirs of N/2 | 63,750 | +0.80 | +0.44 | +0.50 |
| **Selective tuned** | `g(t) = g + G·u(t)` | 2,000 | **−0.23** | −0.03 | +0.10 |
| Tuned + per-neuron gains | `diag(d_L)·U·diag(σ⊙g)·Vᵀ·diag(d_R)` | 2,500 | **−0.13** | −0.01 | −0.02 |
| Tuned + penalty | `λ·mean((g − 1)²)` | 1,500 | 0.00 | 0.00 | 0.00 |
| Tuned + 32 rotations | rotations between pairs of directions | 1,532 | −0.03 | +0.01 | −0.01 |
| *Control: lr of the gains × 0.3 / × 3* | | 1,500 | +0.24 / −0.07 | +0.10 / +0.05 | **−0.07** / +0.01 |

![Variants, 500 neurons](../results/figures/improvements_n500.png)

- **No variant is a clear improvement everywhere**; the plain tuned reservoir stays within 1.7 times of its best variant on
  each task.
- **Two real but targeted gains**: the selective tuned reservoir on Mackey-Glass (1.7 times), the per-neuron gains on
  Mackey-Glass (1.35 times). The selective tuned reservoir also won on NARMA10 at 200 neurons (1.5 times), no longer at 500.
- **What hurts**: tuning the eigenvalues instead of the singular values (the same result comes back in part 6), reducing the
  profile to 8 coefficients, stacking two half-reservoirs.
- **Combining the levers of parts 1 and 2** (tuned + sculpted + braked, at 100 and 500 neurons) brings nothing: no
  combination beats the best single model of each task, and adding braking always degrades. The only notable gain comes from
  the mask on the inputs (fully tuned: 2.2 times better on memory at 500 neurons), which already foreshadowed the role of the
  inputs seen in 4.2.

## 4.5 Tuned gates

The GRU and the LSTM add **gates**, at the price of three or four learned N×N matrices. This path is retraced starting from
the tuned reservoir: the gates are kept, but each recurrent matrix becomes a tuned matrix, with the frozen random directions of
the reservoir and one gain profile per matrix.

| Model | Cell | Parameters (excluding readout) |
|---|---|---|
| **Gated tuned reservoir** | GRU-like (`z`, `r`, `h̃`), with tuned `W, W_z, W_r` | 9 N (against 3 N² for a GRU) |
| Gated tuned reservoir, light gates | neuron-by-neuron gates | 9 N |
| Gated tuned reservoir, LSTM-like | tuned `f, i, o` gates | 12 N |
| **Tuned reservoir with an input gate** | a single gate, which depends only on the input (like minGRU) | 5 N |

![Gates, linear recurrence, fast reservoir](../results/figures/schema_ameliorations.png)

All of them start almost exactly from the tuned reservoir (gates almost open at the start).

![Gallery of the gated tuned reservoir](../results/figures/gated_gallery.png)

The gates stay close to their start but move with the input: the update gate stays at 0.95 on average, and closes down to 0.6
on a few neurons, in rhythm with u(t). The tuned LSTM keeps its forget gate almost closed.

**At 500 neurons, 1,500 steps, 3 seeds** (factor of error reduction relative to the tuned reservoir):

| N = 500 | Parameters | Mackey-Glass | NARMA10 | Memory |
|---|---|---|---|---|
| Tuned reservoir (reference) | 1,500 | 7.5e-8 | 0.0102 | 0.00186 |
| Selective tuned | 2,000 | **×1.70** | ×1.07 | ×0.79 |
| **Gated tuned reservoir** | 4,500 | ×1.02 | **×1.20** | **×1.15** |
| Tuned reservoir with an input gate | 2,500 | ×1.10 | ×1.12 | ×1.05 |
| Gated tuned reservoir, LSTM-like | 6,000 | ×1.12 | ×0.89 | ×0.81 |
| Control: gates only (W = W₀ frozen) | 4,000 | ×0.38 | ×0.65 | ×0.89 |

![Gates, 500 neurons](../results/figures/gated_n500.png)

- Gates improve the tuned reservoir, but little (1.2 times at best), and less at 500 than at 200 neurons.
- **Training only the gates is not enough**: with the main matrix frozen, the model does 1.1 to 2.6 times worse than the
  tuned reservoir without gates. This is the difference with *Gated Echo State Networks*, where training only the gates is
  effective.
- **At 4,000 steps, against the best existing models** (3 seeds): the selective tuned reservoir does 1.6 times better than the
  well-initialized GRU on Mackey-Glass; on NARMA10 and memory, the gated variants do at most 12 % better than the plain tuned
  reservoir. On twelve synthetic tasks (part 8.3), gates only matter on Hénon (1.8 times) and on the Wiener-Hammerstein system
  (1.2 times), and hurt on memory at 30 steps.

## 4.6 The linear tuned reservoir: the full LRU with 200 times fewer parameters

The LRU (Orvieto et al., 2023) removes the nonlinearity from the recurrence: `h(t) = λ ⊙ h(t−1) + γ ⊙ B·u(t)`, then
`GELU(Re(C·h) + D·u)`, and learns λ, B, C and D (about N² parameters). The **linear tuned reservoir** keeps C and D drawn at
random and frozen; only the eigenvalues λ, the normalization γ and the inputs B learn.

| N = 500, 1,500 steps | Parameters | Mackey-Glass | NARMA10 | Memory |
|---|---|---|---|---|
| Full LRU (best of two lr) | 251,750 | **1.0e-4** | 0.0135 | 0.0067 |
| **Linear tuned reservoir** | **1,250** | 1.7e-4 | **0.0129** | **0.0056** |
| Linear tuned, 2 layers | 2,000 | 4.3e-4 | 0.032 | 0.026 |
| *Tuned reservoir (nonlinear)* | 1,500 | *7.5e-8* | *0.0102* | *0.0019* |

![Gallery of linear tuned recurrences](../results/figures/linear_gallery.png)

- **It matches the full LRU** with 200 times fewer parameters (tie on NARMA10, 1.2 times better on memory, 1.7 times worse on
  Mackey-Glass): learning the input and output matrices of the LRU brings almost nothing on these tasks.
- The full LRU mostly moves its outputs C; the linear tuned reservoir, which cannot touch them, moves its eigenvalues and
  **equalizes the memory of the modes** around the useful horizon of the task.
- **But a linear recurrence is far below a nonlinear tuned reservoir** on these short sequences (2,000 times more error on
  Mackey-Glass), and a second layer makes everything worse.
- **On long sequences** (pixel-by-pixel MNIST, 784 steps, 128 neurons, 3 epochs), the LRU wins (28.3 % error), the linear
  tuned reservoir is 6 points behind with 52 times fewer parameters (34.7 %), the nonlinear tuned reservoir beats the GRU and
  the RNN (46.7 % against 59.3 and 64.9 %). On the UEA archive, the linear tuned reservoir again matches the LRU (83.0 %
  against 83.4 % mean accuracy, 11 times fewer parameters).

## 4.7 Size: from 100 to 16,000 neurons

From 100 to 2,000 neurons (3 seeds), against the best RNN, rank 4 and the fully sculpted reservoir:

| Tuned (best of the four models in bold) | N = 100 | 200 | 500 | 1,000 | 2,000 |
|---|---|---|---|---|---|
| Mackey-Glass | 8.8e-5 | **3.6e-6** | **7.4e-8** | 3.4e-8 | 3.4e-8 |
| NARMA10 | 0.023 | 0.016 | 0.0102 | **0.0105** | **0.017** |
| Memory | 0.0029 | 0.0018 | 0.0019 | 0.0020 | 0.0044 |

![Scaling](../results/figures/scaling.png)

- **Each model has its own size range.** On Mackey-Glass, the tuned reservoir plateaus from 1,000 neurons (3.4e-8), while
  rank 4 and the sculpted reservoir keep going down to 3e-9; N gains are enough up to about 500 neurons, not beyond. On
  NARMA10, it is the opposite: the tuned reservoir becomes the best as it grows, because it overfits less than the others. On
  memory, the sculpted reservoir is first at every size.
- **The fast tuned reservoir** replaces the random directions by Fourier directions (`W = P₁·F⁻¹·diag(λ ⊙ g)·F·P₂`),
  applicable in N·log N. Sixteen thousand neurons train in 2 min 30 with 40,000 learned parameters (an RNN of this size would
  have 256 million), but growing does not improve the error on these tasks, which are limited by their 3,000 training points.

![Fast tuned reservoir](../results/figures/fast_scaling.png)

## 4.8 One reservoir, several tasks

The reservoir never changes: one can therefore keep **a single `W₀` for several tasks**, and store, for each task, only what
was learned. Switching tasks amounts to switching settings, like stations on the same radio set.

| 500 neurons, test NMSE | What is stored | Mackey-Glass | NARMA10 | Memory |
|---|---|---|---|---|
| **Tuned, one setting per task** (shared `W₀`) | **2,001 numbers per task** | **7.4e-8** | **0.0104** | **0.0019** |
| Tuned, shared and frozen inputs | 1,501 numbers per task | 1.9e-7 | 0.023 | 0.056 |
| *Best RNN, one copy per task* | *251,501* | *1.1e-6* | *0.093* | *0.036* |
| RNN learned one task after the other (at the end) | a single model | 0.91 | 1.11 | 0.55 |
| Tuned, a single setting learned one task after the other | a single model | 8.9 | 0.82 | 0.0027 |

![One reservoir, several tasks](../results/figures/multitask_n500.png)

- A shared reservoir and one stored setting per task beat one RNN per task (15, 9 and 19 times), for 125 times less to store,
  and nothing is forgotten. At 100 neurons, they match it for 25 times less to store.
- **Sharing the inputs is costly** (up to 30 times on memory): a task needs its own input weights, which echoes 4.2.
- Learning the tasks one after the other in a single model causes **catastrophic forgetting**, for the RNN as for the tuned
  reservoir: having few parameters does not protect from forgetting; storing the settings separately does.

## 4.9 At equal parameter budget: the rank-r correction

Comparing a GRU of 750,000 parameters with a tuned reservoir of 1,500 says nothing without the full curve. The tuned reservoir
is given more freedom (a rank-r correction `U·Vᵀ`, from 1,500 to 257,500 parameters at 500 neurons) and the number of units
of the GRU and the LRU is varied (from 20 to 500).

![Equal-budget curves](../results/figures/budget_curves.png)

| Budget | Model (parameters) | Mackey-Glass | NARMA10 | Memory | Lorenz | Hénon |
|---|---|---|---|---|---|---|
| ≈ 1.4 k | tuned, rank 0 (1,500) | **7.5e-8** | **0.0102** | **0.0019** | **4.3e-5** | 0.065 |
| | GRU 20 units (1,380) | 3.3e-4 | 0.103 | 0.163 | 1.2e-3 | **0.032** |
| ≈ 10 k | tuned, rank 8 (9,500) | **1.5e-7** | 0.0158 | **0.0015** | **3.6e-6** | **0.0034** |
| | GRU 60 units (11,340) | 6.4e-5 | **0.0063** | 0.0045 | 4.3e-5 | 0.028 |
| ≈ 32 k | tuned, rank 32 (33,500) | **1.3e-6** | 0.041 | **0.0023** | 4.4e-6 | **0.0017** |
| | GRU 100 units (30,900) | 1.4e-5 | **0.019** | 0.545 | 6.5e-6 | 0.021 |
| ≈ 125 k | tuned, rank 128 (129,500) | 4.3e-6 | 0.077 | **0.0075** | 1.5e-5 | **0.0022** |
| | GRU 200 units (121,800) | **9.3e-7** | 0.132 | 0.631 | **3.7e-7** | 0.0037 |
| ≈ 260 k | tuned, rank 256 (257,500) | unstable | 0.099 | **0.029** | unstable | 0.016 |
| | GRU 300 units (272,700) | **2.5e-7** | 0.156 | 0.62 | **4.9e-7** | **0.0011** |

- **On Lorenz and Hénon, the rank-r tuned reservoir beats the GRU up to about 100,000 parameters** (up to 12 times: Lorenz at
  10,000 parameters, Hénon at 30,000); beyond that, the GRU takes the lead again. Elsewhere, the rank correction degrades: on
  Mackey-Glass and NARMA10, at every rank; on memory, from r = 32. Rank 256 is unstable (single learning rate).
- **The GRU is bad on NARMA10 and memory only at certain sizes.** A GRU of 60 units beats the tuned reservoir on NARMA10
  (0.0063 against 0.0102) and comes close on memory; at 20, 100 or 500 units, it fails. The GRU therefore has a narrow size
  range where it succeeds on these tasks, which the fair search (part 8.2), drawing the size at random, did not find.
- **And without g?** Since g hurts on Lorenz and Hénon (4.2), the gains are removed and the biases, `W_in` and the rank-r
  correction are kept (3 seeds):

  | Without g, rank r (parameters) | Lorenz | Hénon | Wiener-Hammerstein |
  |---|---|---|---|
  | rank 8 (9,000) | **2.6e-6** (×1.4 against the rank-8 tuned; **×17 against the 60-unit GRU**) | **0.0017** (×2.2; **×17**) | 0.0016 (×1.3; GRU ×2.5 ahead) |
  | rank 32 (33,000) | 4.4e-6 (equal; ×1.7 against the 100-unit GRU, within noise) | **0.0011** (×1.5; **×16**) | 0.0087 (×1.4; GRU ×15 ahead) |
  | rank 128 (129,000) | diverges on one seed | 0.0017 (×1.3; ×2.7 against the 200-unit GRU) | 0.043 (equal; GRU ×85 ahead) |

  On the two chaotic systems, removing g helps a little (1.3 to 2.2 times), and the model beats the GRU of comparable budget
  by 16 to 17 times up to 33,000 parameters; on Hénon, 0.0011 with 33,000 parameters matches the 300-unit GRU (0.0010,
  272,700 parameters). On Wiener-Hammerstein, the GRU wins from 10,000 parameters on and the rank correction overfits
  (30 times more error at rank 128 than at rank 0).

## 4.10 Assessment of part 4

**Where it works.**
- Mackey-Glass and NARMA10 at 500 neurons: with 1,500 parameters, better than the GRU and the LRU tuned with the same search
  budget (1.8 and 1.7 times), and far ahead of the RNN.
- The Silverbox physical system: on par with the GRU, with 500 times fewer parameters, and more stable from one seed to the
  next; the simulated power amplifier and channel equalization (part 8.3), especially when data are scarce.
- Per-task settings stored on a shared reservoir; the linear tuned reservoir, which matches the full LRU.
- Lorenz and Hénon **at equal budget**: with a rank-r correction and without the gains, 16 to 17 times better than the GRU of
  the same size up to 33,000 parameters.

**The limits.**
- **The name partly lies, on synthetic series**: most of the gain there comes from the biases and the inputs, learned in a
  reservoir whose recurrence does not change; well tuned, this reservoir matches the tuned one except on NARMA10, and g hurts
  on Lorenz and Hénon. On real physical systems, on the contrary, the gains are indispensable (1.7 to 3 times less error with
  them).
- Memory: the well-tuned LRU does 6 times better (part 8.2). Lorenz, Hénon and the Wiener-Hammerstein system: the large GRU
  dominates (77 to 190 times on the two chaotic systems against the plain tuned reservoir, 3 times on the real
  Wiener-Hammerstein system, where the GRU also wins at equal budget).
- No gain in classification (MNIST, UEA). The model plateaus on Mackey-Glass beyond 1,000 neurons.

**The link with theory.**
- *Singular value adaptation.* Freezing random matrices and learning only scaling vectors is the principle of
  [VeRA](https://arxiv.org/abs/2310.11454); learning only the singular values of a pretrained weight is that of SVFit
  ([arXiv:2409.05926](https://arxiv.org/abs/2409.05926)), [SVFT](https://neurips.cc/virtual/2024/poster/93272),
  [SVDiff](https://arxiv.org/abs/2303.11305) and Transformer² ([arXiv:2501.06252](https://arxiv.org/abs/2501.06252)).
  Parametrizing an RNN by its decomposition `U·Σ·Vᵀ` is known ([Zhang, Lei and Dhillon, 2018](https://arxiv.org/abs/1803.09327)),
  with `U` and `V` learned. The tuned reservoir applies this mechanism to the recurrence of a random reservoir: the transfer is
  new, not the mechanism.
- *Intrinsic plasticity.* Adapting a gain and a bias per neuron of a reservoir has been known since
  [Steil (2007)](https://www.esann.org/sites/default/files/proceedings/legacy/es2007-98.pdf) and Schrauwen et al. (2008), with
  a local unsupervised rule. The result of 4.2 (the biases carry a large share of the gain) places the tuned reservoir very
  close to this line; the difference is the criterion (supervised, by gradient) and the learning of the inputs.
- *Why the biases matter.* For a `tanh`, the bias shifts the operating point of each neuron: it sets its local slope, hence
  the effective amplification of the linearized dynamics (`diag(1 − x²)·W`), and breaks the odd symmetry of the states of a
  bias-free reservoir. Learning the biases therefore amounts to tuning the dynamics **per neuron** rather than per singular
  direction; this is an interpretation, not a proof.
- *Why g lengthens memory.* The memory capacity of an almost linear network depends on how `W` amplifies and damps each
  direction: increasing the singular values of a few directions makes them slower, hence a longer memory (31 against 22 for
  the RNN). On Lorenz and Hénon, where products of variables are needed above all, this freedom is of no use and adds
  overfitting.
- *Gates and linear recurrences.* Gated ESNs ([Di Sarli, Gallicchio and Micheli, 2022](https://zenodo.org/record/7040474))
  train only the gates; minGRU ([Feng et al., 2024](https://arxiv.org/abs/2410.01201)) inspired the input gate; a diagonalized
  linear ESN with sampled eigenvalues ([arXiv:2602.19802](https://arxiv.org/abs/2602.19802)) has the structure of the linear
  tuned reservoir, without training its eigenvalues. Per-task adapters on a shared reservoir take up the spirit of LoRA and
  SVDiff. The rank-r correction on a frozen random base is the principle of LottaLoRA (part 8.4).

---

[Contents](../README.en.md#the-report) · [← Part 3](3_tourne.md) · [Part 5 →](5_cisaille.md)
