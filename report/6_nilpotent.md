[Contents](../README.en.md#the-report) · [← Part 5](5_cisaille.md) · [Part 7 →](7_couplage.md) · [Français](../rapport/6_nilpotent.md)

# Part 6 — The nilpotent reservoir: eigenmodes and nilpotent part (Jordan-Chevalley)

![Nilpotent reservoir](../results/figures/schema_nilpotent.png)

> **What to remember**
>
> - The reservoir is split into a semisimple part (its eigenmodes, whose modulus and phase are learned) and a learned nilpotent part, which changes no eigenvalue but creates transient amplification.
> - Tuning only the eigenmodes is **worse** than tuning the singular values (2.2 to 2.6 times on memory). Adding the nilpotent part gains a lot on synthetic tasks (Mackey-Glass ×2, Lorenz ×4.3) but loses on Hénon.
> - At 500 neurons, the full version (125,000 parameters) collapses on four tasks out of five and **degrades all five tests on real systems**. The banded version keeps a gain on Lorenz (3.3 times) but loses on Hénon (4.6 times): it stays clearly below the banded sheared reservoir (part 5).

The Jordan-Chevalley decomposition writes any matrix as the sum of a **semisimple** part `S` (diagonalizable) and a
**nilpotent** part `N` (`Nᵏ = 0`), which commute. The nilpotent part changes no eigenvalue, but it moves energy from one mode
to another: a signal is amplified for a few steps before dying out.

| Model | Form | What is learned | Parameters at 100 neurons |
|---|---|---|---|
| Tuned on the eigenvalues (semisimple part) | `W = Re(P·diag(λ ⊙ g·e^{iφ})·P⁻¹) + N₀` | modulus `g` and phase `φ` of each mode | 400 |
| **Nilpotent reservoir** | same `+ Qr·T·Qrᵀ` | the modes and `T`, strictly triangular | 5,350 |

A dense random matrix is almost surely diagonalizable: its nilpotent part `N₀` is zero. A reservoir with 10 % of connections,
on the other hand, has Jordan blocks; `N₀ = W₀ − S₀` is then computed and kept frozen. The learned part `Qr·T·Qrᵀ` (`Qr` a
frozen random orthogonal matrix) is nilpotent, but its commutation with the semisimple part, required by the theorem, **is not
imposed**: it is a model inspired by the decomposition, not the exact decomposition.

## 6.1 Results at 100 neurons

Five tasks, 3 seeds, error factor relative to the tuned reservoir (×2 = half the error):

| Model | Parameters | Mackey-Glass | NARMA10 | Memory | Lorenz | Hénon |
|---|---|---|---|---|---|---|
| Tuned reservoir (singular values), NMSE | 300 | 9.0e-5 | 0.023 | 0.0029 | 6.3e-4 | 0.042 |
| Tuned on the eigenvalues | 400 | ×1.1 | ÷1.3 | **÷2.6** | ÷1.8 | ×1.0 |
| **Nilpotent reservoir** | 5,350 | **×2.0** | ÷1.0 | **×1.3** | **×4.3** | **÷2.2** |
| *Part 5: sheared reservoir* | 5,150 | ×1.2 | **×1.2** | **×1.2** | **×4.2** | **×2.0** |
| *RNN* | 10,200 | ×1.1 | **÷1.4** | ×1.0 | **×4.7** | ÷1.6 |

- **Eigenmodes are a worse lever than singular values**: at a similar budget, they lose 2.6 times on memory. The same gap is
  found at 500 neurons (2.2 times on memory, 2 times on NARMA10; part 4.4).
- **The nilpotent part gains a lot on synthetic tasks**: 2 times on Mackey-Glass (more than the sheared reservoir), 4.3 times
  on Lorenz, almost as much as the RNN. But it **loses on Hénon** (2.2 times), where the sheared reservoir wins.

## 6.2 At 500 neurons, full and banded

Five tasks, 3 seeds; error factor relative to the tuned reservoir (1,500 parameters):

| Model (parameters) | Mackey-Glass | NARMA10 | Memory | Lorenz | Hénon |
|---|---|---|---|---|---|
| Full nilpotent reservoir (126,750) | **÷118** | **÷11** | **÷47** | **×4.1** | **÷4.7** |
| Banded nilpotent reservoir, k = 3 (3,494) | equal | **÷1.4** | **÷1.3** | **×3.3** | **÷4.6** |
| *Part 5: banded shear, k = 3 (2,494)* | ÷1.2 | **×1.6** | **×1.5** | **×6.2** | **×1.5** |

Like the sheared reservoir, the full version collapses at this size, except on Lorenz. The band avoids the collapse but only
wins on Lorenz; on Hénon, the nilpotent part loses as much banded as full. At equal structure (a band of three
superdiagonals), the shear does better than the nilpotent part on all five tasks.

## 6.3 On real physical systems

At 500 neurons (3 seeds), the nilpotent part costs about 125,000 parameters. On the three real systems of part 8.3, paired gap
in log10 of the RMSE relative to the tuned reservoir (+0.30 = twice the error):

| | Silverbox, multisine | Silverbox, arrow with extrapolation | Silverbox, arrow without extrapolation | Wiener-Hammerstein | Cascaded Tanks |
|---|---|---|---|---|---|
| Tuned reservoir (1,500 parameters), RMSE | 4.3 mV | 9.9 mV | 4.0 mV | 11.8 mV | 1.69 V |
| Nilpotent reservoir (126,750), gap | +0.36 | +0.19 | +0.38 | +0.21 | +2.24 (diverges) |

The nilpotent part **degrades all five tests** (1.5 to 2.4 times more error, divergence on Cascaded Tanks, which has only 1,024
training points). Overfitting is the most plausible explanation, consistent with the collapse on synthetic tasks at the same
size. The banded version was not measured on these systems, for lack of a synthetic gain outside Lorenz.

## 6.4 Assessment of part 6

**Where it works.** Mackey-Glass (2 times better than the tuned reservoir) and Lorenz (4.3 times) at 100 neurons; Lorenz only
at 500 neurons (3.3 times, banded).

**The limits.**
- Loss on Hénon (2.2 times at 100 neurons, 4.6 times at 500), whereas the sheared reservoir wins there.
- Collapse of the full version at 500 neurons, on synthetic tasks as well as on real systems.
- Banded, always worse than the banded sheared reservoir of the same size.
- The commutation required by the theorem is not imposed; 100 neurons and 3 seeds for the synthetic part.

**The link with theory.**
- *Eigenvalues against singular values.* Eigenvalues set the **asymptotic** behaviour (stability, long-term decay time);
  singular values set the amplification **at each step**. For a memory of 15 steps, short-term amplification is what counts,
  which is consistent with the advantage of singular values (part 4) over eigenmodes. For a non-normal matrix, the two can be
  very different, and this is exactly what the nilpotent part exploits.
- *Transient amplification.* With `W = S + N`, `Wᵗ = Σₖ C(t, k)·Sᵗ⁻ᵏ·Nᵏ` when `S` and `N` commute: the terms in `N` add powers
  of t in front of the modes (the behaviour of a Jordan block), hence a polynomial growth before the exponential decay. This is
  the transient memory of non-normal networks (Ganguli et al., 2008).
- *Related work.* The nnRNN ([Kerg et al., 2019](https://arxiv.org/abs/1905.12080)) learns a triangular part in the real Schur
  form `Q·(Λ + T)·Qᵀ`, the numerically stable version of this idea; the nilpotent reservoir is a variant of it applied to a
  reservoir, with `Q` and the spectrum frozen. It is not a new idea.

---

[Contents](../README.en.md#the-report) · [← Part 5](5_cisaille.md) · [Part 7 →](7_couplage.md)
