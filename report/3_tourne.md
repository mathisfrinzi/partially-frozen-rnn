[Contents](../README.en.md#the-report) · [← Part 2](2_sculpte.md) · [Part 4 →](4_accorde.md) · [Français](../rapport/3_tourne.md)

# Part 3 — The rotated reservoir: learning only the orientation

![Rotated reservoir](../results/figures/schema_tourne.png)

> **What to remember**
>
> - The spectrum of the reservoir (eigenvalues and singular values) is kept exactly, and only a rotation `W = Q·W₀·Qᵀ` is learned.
> - It helps chaotic prediction tasks (Mackey-Glass, and above all Hénon) but **fails on memory** (10 times worse than the RNN) and on MNIST: unable to change its gains, the network cannot lengthen its memory.
> - It is the "directions" half of a question whose "gains" half is studied in part 4.

Does the quality of a reservoir come from its **spectrum** (the eigenvalues and singular values, which set stability and
amplification) or from its **directions** (the orientation of the dynamics in neuron space)? To find out, one is frozen and
the other learned. The rotated reservoir freezes the spectrum:

```
W = Q · W₀ · Qᵀ,     Q = exp(S − Sᵀ) orthogonal,  starting from Q = I
```

`Q·W₀·Qᵀ` has exactly the same eigenvalues and the same singular values as `W₀`. Two variants: the inputs rotate with the
recurrence (`W_in ← Q·W_in`), or stay free.

## 3.1 At 100 neurons: a gain in prediction, a failure in memory

Paired gap to the RNN (standard learning rate), 8 seeds (5 for memory):

| | Mackey-Glass | NARMA10 | Memory |
|---|---|---|---|
| RNN | 0.000085 | 0.0316 | 0.0027 |
| Part 1: braked RNN | 0.000032 | 0.0082 | 0.0011 |
| **Rotated reservoir** (inputs rotated with `Q`) | **0.000047** (−0.24 ± 0.05) | 0.0266 (−0.09 ± 0.06) | 0.0297 (**+1.01** ± 0.08) |
| Rotated reservoir (free inputs) | 0.000072 (−0.07 ± 0.08) | 0.0250 (−0.10 ± 0.05) | 0.0329 (+1.06 ± 0.08) |
| Part 4: tuned reservoir (gains only) | 0.000084 (0.00 ± 0.09) | 0.0242 (−0.11 ± 0.04) | **0.0028** (+0.02 ± 0.03) |

![Spectrum against directions](../results/figures/rotation.png)

- **Memory**: keeping the spectrum and learning only the orientation fails (11 times worse than the RNN), while learning
  only the gains (tuned) matches the RNN. To remember, the amplification must change.
- **Mackey-Glass**: it is the opposite. Reorienting the dynamics beats the RNN by 1.8 times; changing only the gains brings
  nothing.
- **NARMA10**: both do a little better than the RNN, with no difference between them.

## 3.2 On chaotic systems built from products

On five tasks at 100 neurons (3 seeds, PyTorch), error factor relative to the tuned reservoir (×2 = half the error) and to
the RNN:

| Rotated reservoir | Mackey-Glass | NARMA10 | Memory | Lorenz | Hénon |
|---|---|---|---|---|---|
| against the tuned reservoir | ×1.3 | ×1.0 | **÷9.3** | ×1.7 | **×2.8** |
| against the RNN | ×1.8 | ×1.2 | ÷10 | ÷2.7 | **×4.4** |

On Hénon (a quadratic map), reorienting the dynamics clearly pays off: it is the best of the non-coupled models at 100
neurons on this task. On Lorenz, it stays behind the RNN.

## 3.3 Seen from the inside: the shortest memory

Memory capacity (Jaeger, 2002; part 4.3 for the details of the measurement): at 100 neurons, trained on the memory task, the
rotated reservoir keeps the shortest memory of all models (19.5, against 22.5 for the RNN and 31 for the tuned reservoir) and
almost no nonlinear capacity (0.6). Trained on Mackey-Glass, on the contrary, it has the longest memory of the trained models
(11.2, almost that of the starting reservoir, 12.1): it does not sacrifice it to compute. Unable to change its spectrum, it
cannot lengthen its memory: this is its failure of table 3.1, seen from the inside.

## 3.4 Larger, and on non-synthetic data

- **500 neurons** (3 seeds, Mackey-Glass and NARMA10): it beats the best RNN (6.6 and 2.4 times), but it is the weakest of the
  constrained models at this size.
- **Row-by-row MNIST** (256 neurons): 11.65 % error, against 4.4 % for the RNN and 5.1 % for the tuned reservoir. Reorienting
  the dynamics without changing its gains is not enough to build representations.
- **Only a few rotations.** Adding 32 learned rotations between pairs of directions to a tuned reservoir (part 4.4) changes
  nothing (gaps of 0.03 at most): a little orientation at very low cost does not help.

## 3.5 Assessment of part 3

**Where it works.** Prediction of chaotic systems: Mackey-Glass (1.8 times better than the RNN at 100 neurons) and above all
Hénon (4.4 times better than the RNN, 2.8 times better than the tuned reservoir).

**The limits.**
- Failure on memory (10 times worse than the RNN) and on MNIST.
- The rotation costs one matrix exponential per learning step (O(N³)), which limits the testable size; it learns N(N−1)/2
  parameters, without being more frugal than an RNN.
- No measurement on real physical systems.

**The link with theory.**
- *Why memory cannot come from a rotation.* If the inputs rotate with the recurrence, a **linear** network
  `x ← Q·W₀·Qᵀ·x + Q·W_in·u` is exactly the network `W₀` seen in another basis (`x' = Qᵀ·x`): a linear readout recovers the
  same information. The rotation therefore changes nothing to what an almost linear regime can retain; it only acts through
  the nonlinearity, applied coordinate by coordinate, which is not rotation-invariant. Linear memory, for its part, is set by
  the spectrum and the alignment of the inputs (Jaeger, 2002; Ganguli et al., 2008). This is consistent with the failure on
  memory and the gain on tasks where the nonlinearity is the heart of the problem (Hénon).
- *Orthogonal and unitary RNNs.* Parametrizing a rotation by the exponential of a skew-symmetric matrix is the method of
  orthogonal RNNs ([Lezcano-Casado and Martínez-Rubio, 2019](https://arxiv.org/abs/1901.08428); Mhammedi et al., 2017;
  unitary RNNs, [Arjovsky et al., 2016](https://arxiv.org/abs/1511.06464)). Here, the rotation is not the recurrence itself:
  it serves as a change of basis around a reservoir whose spectrum is frozen. The dissociation "spectrum for memory,
  orientation for prediction", measured by freezing one or the other on the same reservoir, does not seem to have been tested
  as such.

---

[Contents](../README.en.md#the-report) · [← Part 2](2_sculpte.md) · [Part 4 →](4_accorde.md)
