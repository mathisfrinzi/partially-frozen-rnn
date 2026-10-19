[Contents](../README.en.md#the-report) · [← Part 6](6_nilpotent.md) · [Part 8 →](8_synthese.md) · [Français](../rapport/7_couplage.md)

# Part 7 — Coupling a frozen reservoir and a learned RNN

![Four forms of coupling](../results/figures/schema_couplage.png)

> **What to remember**
>
> - **Cross duos** (a reservoir and an RNN that read each other with a delay) bring nothing, and diverge at 500 neurons.
> - **Relays** (a memoryless layer inside the transition) gain a lot on Hénon (9 times better than the GRU at 500 neurons), but a fully learned relay does as well: what matters is the **depth of the transition**, not randomness. With a tuned block in the transition, the gain disappears.
> - A **squared readout** `[x, x²]` on the tuned reservoir captures a good part of the gain without coupling, but it is mostly a wider reservoir, except on Lorenz and Hénon (polynomial features, already known from NVAR). None of this transfers to real systems.

Parts 1 to 6 transform a single reservoir. Here, **two distinct objects** are kept, a frozen random reservoir and a fully
learned RNN, and they are made to read each other: it is the most literal version of "reservoir + RNN".

## 7.1 The four forms

A second state `y` is introduced: one of the two is a **reservoir** (`R`, frozen weights), the other an **RNN** (`A`, learned
weights). Two choices: which one is in the time loop, and whether the second block is read with a delay (parallel update) or
without (in cascade).

| Form | Name | Equations | Remark |
|---|---|---|---|
| 1 | **Cross duo, reservoir first** | `y(t+1) = f_R(W_inR u + W_R y + C x)`, `x(t+1) = f_A(W_inA u + W_A x + D y)` | two delayed, coupled states |
| 2 | **Cross duo, RNN first** | roles swapped | equivalent to 1 if the readout reads both states |
| 3 | **Random relay** | `y(t) = g(W_inR u(t) + W_R x(t))`, `x(t+1) = f(W_in u(t) + W_A y(t))` | a frozen random memoryless layer inside the learned transition |
| 4 | **Learned relay** | `y(t) = g(W_inA u(t) + W_A x(t))`, `x(t+1) = f(W_inR u(t) + W_R y(t))` | roles swapped |

Controls: the **fully learned relay** (both layers learned: a deep-transition RNN, DT-RNN) and the **squared readout**
(`[x, x²]`, without coupling).

## 7.2 Results at 100 neurons

Test NMSE, 3 seeds; parameters of the recurrent part.

| Model | Parameters | Memory | NARMA10 | Mackey-Glass | Lorenz | Hénon |
|---|---|---|---|---|---|---|
| Classic reservoir | 0 | 0.79 | 0.28 | 0.0017 | 0.0023 | 0.67 |
| RNN | 10,200 | **0.0028** | 0.032 | 8.0e-5 | 2.8e-4 | 0.078 |
| Tuned reservoir | 300 | 0.0029 | 0.023 | 9.0e-5 | 6.3e-4 | 0.042 |
| GRU, careful init | 30,900 | 0.51 | **0.0125** | **1.2e-5** | **8.3e-6** | 0.031 |
| Form 1: cross duo, read on the RNN side | 30,200 | 0.032 | 0.088 | 1.0e-4 | 2.8e-4 | 0.26 |
| Form 2: cross duo, read on the reservoir side | 30,200 | 0.049 | 0.13 | 7.6e-5 | 7.8e-5 | 0.75 |
| Form 3: random relay | 10,200 | 0.023 | 0.036 | 1.0e-4 | 9.9e-5 | **0.00099** |
| Form 4: learned relay | 10,200 | 0.024 | 0.033 | 9.9e-5 | 1.2e-4 | 0.0027 |
| *Fully learned relay (DT-RNN)* | 20,300 | 0.012 | 0.072 | 1.7e-4 | 3.6e-4 | **0.00049** |
| *Tuned reservoir, squared readout* | 300 | 0.0037 | 0.0145 | 1.1e-5 | 8.4e-5 | 0.013 |

- **Cross duos: nothing.** Worse than the tuned reservoir on memory, NARMA10 and Hénon, with three times more parameters.
- **Relays: a clear gain on chaotic systems built from products.** On Hénon, form 3 does 43 times better than the tuned
  reservoir and 32 times better than the GRU; on Lorenz, 5 to 6 times better than the tuned reservoir (the GRU stays 12 times
  ahead). Elsewhere, equal or worse.
- **But the fully learned relay does as well** (0.00049 on Hénon): the gain comes from the **depth of the transition** (two
  `tanh` per step), not from randomness. In favour of form 3: its frozen layer costs half as many parameters.
- **The squared readout captures a good part of the gain without coupling**: on the tuned reservoir, it goes from 0.042 to
  0.013 on Hénon, from 6.3e-4 to 8.4e-5 on Lorenz, from 9.0e-5 to 1.1e-5 on Mackey-Glass.

## 7.3 At 500 neurons: a question of learning rate

| Model (3 seeds) | Parameters | Mackey-Glass | NARMA10 | Memory | Lorenz | Hénon |
|---|---|---|---|---|---|---|
| *Tuned + squared readout* | 1,500 | **2.5e-8** | **0.0084** | **0.0018** | 1.7e-5 | 0.023 |
| Tuned reservoir | 1,500 | 7.5e-8 | 0.0102 | 0.0019 | 4.3e-5 | 0.065 |
| GRU, careful init | 754,500 | 1.1e-7 | 0.154 | 0.554 | **2.4e-7** | 8.5e-4 |
| Form 3: random relay (default lr) | 251,000 | 3.9e-5 | 0.18 | 2.76 | 1.5e-4 | **9.0e-5** |
| Form 3: random relay, lr × 0.1 | 251,000 | 2.1e-6 | 0.033 | 0.024 | 1.4e-5 | 2.4e-3 |
| Form 4: learned relay, lr × 0.1 | 251,000 | 3.7e-6 | 0.036 | 0.050 | 1.8e-5 | 3.6e-3 |
| *Fully learned relay (DT-RNN, default lr)* | 501,500 | 0.149 | 0.678 | 1.02 | 0.54 | 1.6e-4 |
| RNN, lr × 0.1 | 251,000 | 2.1e-6 | 0.094 | 0.040 | 6.1e-5 | 0.87 |
| Forms 1 and 2: cross duos | 751,000 | 0.10 to 0.22 | 0.63 to 0.65 | 0.40 | 0.015 to 0.12 | 0.44 to 0.56 |

- The learning rate explains the failure of the relays on four tasks: with lr × 0.1, form 3 gains a factor of 18
  (Mackey-Glass), 5.5 (NARMA10), 115 (memory) and 11 (Lorenz). **On Hénon, it is the opposite**: at the default learning
  rate, the relay reaches 9.0e-5, **9 times better than the GRU**, and the DT-RNN does as well. The right setting depends on
  the task.
- Even well tuned, the relays stay far from the tuned reservoir on Mackey-Glass (28 times), NARMA10 (3.2 times) and memory
  (13 times), with 167 times more parameters. The cross duos always diverge.

## 7.4 Combining: a random relay around a tuned reservoir

The natural combination: keep the random relay, but replace the free RNN of the transition by a tuned reservoir (300
parameters instead of 10,200), with or without the squared readout.

| Model (100 neurons, 3 seeds) | Parameters | Mackey-Glass | NARMA10 | Memory | Lorenz | Hénon |
|---|---|---|---|---|---|---|
| Tuned reservoir | 300 | 9.0e-5 | 0.023 | **0.0029** | 6.3e-4 | 0.042 |
| Tuned, squared readout | 300 | **1.1e-5** | **0.0148** | 0.0037 | 8.4e-5 | 0.013 |
| Form 3: relay + free RNN | 10,200 | 1.0e-4 | 0.035 | 0.024 | 1.1e-4 | 9.5e-4 |
| Form 3, squared readout | 10,200 | 6.9e-5 | 0.052 | 0.025 | 1.4e-4 | **6.4e-5** |
| Relay + tuned RNN | 300 | 1.0e-4 | 0.060 | 0.23 | 6.2e-4 | 0.024 |
| Relay + tuned RNN, squared readout | 300 | 2.0e-5 | 0.048 | 0.33 | **3.2e-5** | 0.0066 |
| Relay + sculpted RNN | 10,200 | 7.8e-5 | 0.15 | 0.80 | 1.6e-4 | 0.0097 |
| GRU, careful init | 30,900 | 1.4e-5 | 0.019 | 0.55 | 6.5e-6 | 0.021 |

- **The relay does not combine with the tuned reservoir**: relay + tuned makes 26 to 32 times more error than form 3 on
  Hénon, and gains nothing on Lorenz. The relay wins because its transition block is a free RNN, not because it is random.
- With the squared readout, it is the best of the 300-parameter models on Lorenz (the GRU stays 5 times ahead), but form 3 +
  squared is 100 times better on Hénon. Relay + sculpted does no better than form 3, and every relay variant is costly on
  memory (8 to 260 times).
- On real systems (part 8.3), relay + tuned + squared readout makes 1.5 to 1.8 times **more** error than the tuned reservoir
  alone, on all five tests.

## 7.5 Why the squared readout works

Three explanations, three controls with the same number of features (2N): (a) there are simply more features (control: a
tuned reservoir with 2N neurons); (b) parity (`tanh` gives odd states, the square completes them; control: the cube
`[x, x³]`, odd); (c) products between different states are needed (control: N products `xᵢxⱼ` drawn at random).

| Factor gained over the tuned reservoir (100 neurons, 3 seeds) | Mackey-Glass | NARMA10 | Memory | Lorenz | Hénon |
|---|---|---|---|---|---|
| `[x, x²]` | ×8.5 | ×1.6 | ÷1.3 | ×7.9 | ×3.5 |
| `[x, x³]` (odd) | ×8.9 | ×1.4 | equal | ×6.5 | ×5.4 |
| `[x, |x|]` | ×1.7 | equal | ÷1.2 | ×1.6 | ×4.5 |
| `[x, xᵢxⱼ]` | ×9.3 | ×1.4 | equal | ×6.3 | ×2.0 |
| **Tuned at 200 neurons**, linear readout | **×24** | ×1.5 | ×1.6 | ×3.2 | ÷1.3 |

- **On Mackey-Glass, NARMA10 and memory, the squared readout is only a way to enlarge the reservoir**: a tuned reservoir with
  2N neurons does as well or better, and it helps memory.
- **On Lorenz and Hénon, it is not a size effect**: nonlinear features beat width. These are two systems made of products;
  it is the result expected from NVAR.
- **Parity is not the cause**: the cube does like the square, including on the bias-free reservoir for Hénon. All products
  are equivalent.
- On memory, no added nonlinearity helps; only width helps.
- **On real systems**, the squared readout changes nothing (gaps of −0.02 to +0.09 in log): the gain is specific to synthetic
  series.

## 7.6 Assessment of part 7

**Where it works.** Hénon (and a little Lorenz): a deep transition, random or learned, beats the GRU on Hénon. The squared
readout helps the tuned reservoir on synthetic series.

**The limits.**
- Relays are costly on memory and NARMA10, require a learning rate specific to each task, and do not combine with the tuned
  reservoir.
- Cross duos bring nothing. The comparison with an additive mixture of the same size was not made.
- Nothing transfers to real systems (squared readout: no effect; relay + tuned: 1.5 to 1.8 times worse).

**The link with theory.**
- *Transition depth.* Deep-transition RNNs (DT-RNN, Pascanu et al., 2014) and their variants with gates between layers
  (GF-RNN) place memoryless layers inside the transition; the random relay is "a DT-RNN with one random layer". A quadratic
  map like Hénon requires products: two successive `tanh` approximate them better than one.
- *Polynomial features.* The squared readout is a minimal version of NVAR / *next-generation reservoir computing* (Gauthier
  et al., 2021) and of the quadratic features of Pathak et al. (2018) for Lorenz; the question of parity comes from
  [*Breaking Symmetries of the Reservoir Equations*](https://arxiv.org/pdf/2010.07103).
- *Existing couplings.* DeepESN and stacked reservoirs (all frozen); GRU-ESN and HBP-ESN (a learned part in the readout only);
  the state-feedback ESN ([arXiv:2312.15141](https://arxiv.org/abs/2312.15141)) and LoRA-RC (a low-rank correction of the
  reservoir, close to the corrected reservoir of part 1); Pathak et al. (2018) couple a physical model and a reservoir; FORCE
  loops a reservoir on its output. No work was found that places a frozen random layer inside the learned transition of an
  RNN, or that compares these four couplings; the search is not exhaustive.

---

[Contents](../README.en.md#the-report) · [← Part 6](6_nilpotent.md) · [Part 8 →](8_synthese.md)
