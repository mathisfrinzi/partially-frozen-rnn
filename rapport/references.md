[Sommaire](../README.md#le-rapport)

# Références

Entre parenthèses, le niveau de lecture quand l'article n'a pas été lu en entier.

## Réservoirs et réseaux récurrents

- H. Jaeger (2001), *The "echo state" approach to analysing and training recurrent neural networks*.
- H. Jaeger (2002), *Short term memory in echo state networks* (capacité mémoire).
- W. Maass, T. Natschläger, H. Markram (2002), *Real-time computing without stable states*.
- D. Sussillo, L. F. Abbott (2009), *Generating coherent patterns of activity from chaotic neural networks* (FORCE).
- M. Lukoševičius, H. Jaeger (2009), *Reservoir computing approaches to recurrent neural network training*, Computer Science Review. [PDF](https://www.ai.rug.nl/minds/uploads/2261_LukoseviciusJaeger09.pdf) (parcouru).
- J. Dambre, D. Verstraeten, B. Schrauwen, S. Massar (2012), *Information processing capacity of dynamical systems*, Scientific Reports.
- T. Strauss, W. Wustlich, R. Labahn (2012), *Design Strategies for Weight Matrices of Echo State Networks*, Neural Computation.
- H. Palangi, L. Deng, R. K. Ward (2013), *Learning Input and Recurrent Weight Matrices in Echo State Networks*. [arXiv:1311.2987](https://arxiv.org/abs/1311.2987)
- C. Gallicchio, A. Micheli (2017), *Deep Echo State Network (DeepESN)*.
- F. M. Bianchi, S. Scardapane, S. Løkse, R. Jenssen (2020), *Reservoir Computing Approaches for Representation and Classification of Multivariate Time Series*, IEEE TNNLS. [arXiv:1803.07870](https://arxiv.org/abs/1803.07870)
- *Breaking Symmetries of the Reservoir Equations in Echo State Networks* (2020). [arXiv:2010.07103](https://arxiv.org/pdf/2010.07103) ; *Asymmetrically connected reservoir networks learn better* (2024). [arXiv:2410.00584](https://arxiv.org/pdf/2410.00584) (résumés)
- K. Cho et al. (2014), *Learning Phrase Representations using RNN Encoder-Decoder* (GRU) ; S. Hochreiter, J. Schmidhuber (1997), *Long Short-Term Memory*.
- A. Orvieto et al. (2023), *Resurrecting Recurrent Neural Networks for Long Sequences* (LRU), ICML. [arXiv:2303.06349](https://arxiv.org/abs/2303.06349)
- L. Feng et al. (2024), *Were RNNs All We Needed?* (minGRU, minLSTM). [arXiv:2410.01201](https://arxiv.org/abs/2410.01201)

## Partie 1 — mélanger, freiner

- G. Yang, E. J. Hu (2021), *Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks*, ICML. [arXiv:2011.14522](https://arxiv.org/abs/2011.14522)
- G. Yang et al. (2022), *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*. [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)
- J. Steil, U. Schiller (2004), rétropropagation-décorrélation (BPDC), dérivée de l'apprentissage récurrent d'Atiya et Parlos. [PDF](https://ni.www.techfak.uni-bielefeld.de/files/SchillerSteil2004-ATW.pdf) (titre seulement)
- Y. H. Liu et al. (2024), *How connectivity structure shapes rich and lazy learning in neural circuits*, ICLR. [arXiv:2310.08513](https://arxiv.org/abs/2310.08513)
- F. Schuessler, F. Mastrogiuseppe, A. Dubreuil, S. Ostojic, O. Barak (2020), *The interplay between randomness and structure during learning in RNNs*, NeurIPS. [arXiv:2006.11036](https://arxiv.org/abs/2006.11036)
- F. Schuessler, F. Mastrogiuseppe, S. Ostojic, O. Barak (2023), *Aligned and oblique dynamics in recurrent neural networks*. [arXiv:2307.07654](https://arxiv.org/abs/2307.07654)
- F. Mastrogiuseppe, S. Ostojic (2018), *Linking connectivity, dynamics and computations in low-rank recurrent neural networks*, Neuron. [arXiv:1711.09672](https://arxiv.org/abs/1711.09672) ; F. Schuessler et al. (2020), *Dynamics of random recurrent networks with correlated low-rank structure*. [arXiv:1909.04358](https://arxiv.org/abs/1909.04358) (résumés)
- *State-feedback echo state networks*. [arXiv:2312.15141](https://arxiv.org/abs/2312.15141) (résumé)

## Partie 2 — sculpter

- H. Zhou et al. (2019), *Deconstructing lottery tickets: zeros, signs, and the supermask*, NeurIPS.
- V. Ramanujan et al. (2020), *What's Hidden in a Randomly Weighted Neural Network?*, CVPR. [arXiv:1911.13299](https://arxiv.org/abs/1911.13299)
- A. Mallya, D. Davis, S. Lazebnik (2018), *Piggyback: Adapting a Single Network to Multiple Tasks by Learning to Mask Weights*, ECCV. [arXiv:1801.06519](https://arxiv.org/abs/1801.06519)
- *Hidden-Fold Networks: Random Recurrent Residuals Using Sparse Supermasks* (2021). [arXiv:2111.12330](https://arxiv.org/abs/2111.12330)
- J. Frankle, D. Schwab, A. Morcos (2021), *Training BatchNorm and Only BatchNorm*, ICLR. [arXiv:2003.00152](https://arxiv.org/abs/2003.00152)
- E. Williams et al. (2025), *Expressivity of Neural Networks with Random Weights and Learned Biases*, ICLR. [arXiv:2407.00957](https://arxiv.org/abs/2407.00957)
- *Rethinking Reservoir Pruning: A Dynamical Perspective for Echo State Networks* (2026). [arXiv:2608.04593](https://www.alphaxiv.org/abs/2608.04593) (résumé)
- *Learning Reservoir Dynamics with Temporal Self-Modulation* (2023). [arXiv:2301.09235](https://arxiv.org/abs/2301.09235) (résumé)
- J. P. Stroud, M. A. Porter, G. Hennequin, T. P. Vogels (2018), *Motor primitives in space and time via targeted gain modulation in cortical networks*, Nature Neuroscience.

## Partie 3 — tourner

- M. Arjovsky, A. Shah, Y. Bengio (2016), *Unitary Evolution Recurrent Neural Networks*, ICML. [arXiv:1511.06464](https://arxiv.org/abs/1511.06464)
- Z. Mhammedi, A. Hellicar, A. Rahman, J. Bailey (2017), *Efficient Orthogonal Parametrisation of Recurrent Neural Networks Using Householder Reflections*, ICML. [arXiv:1612.00188](https://arxiv.org/abs/1612.00188)
- M. Lezcano-Casado, D. Martínez-Rubio (2019), *Cheap Orthogonal Constraints in Neural Networks*, ICML. [arXiv:1901.08428](https://arxiv.org/abs/1901.08428)
- Y. Ma et al. (2024), *Parameter Efficient Quasi-Orthogonal Fine-Tuning via Givens Rotation*, ICML. [PMLR](https://proceedings.mlr.press/v235/ma24a.html) ; *DizzyRNN* (2016). [arXiv:1612.04035](https://arxiv.org/abs/1612.04035) (résumés)

## Partie 4 — accorder

- J. Zhang, Q. Lei, I. Dhillon (2018), *Stabilizing Gradients for Deep Neural Networks via Efficient SVD Parameterization*, ICML. [arXiv:1803.09327](https://arxiv.org/abs/1803.09327)
- E. J. Hu et al. (2021), *LoRA: Low-Rank Adaptation of Large Language Models*.
- L. Han et al. (2023), *SVDiff: Compact Parameter Space for Diffusion Fine-Tuning*, ICCV. [arXiv:2303.11305](https://arxiv.org/abs/2303.11305)
- D. J. Kopiczko, T. Blankevoort, Y. M. Asano (2024), *VeRA: Vector-based Random Matrix Adaptation*, ICLR. [arXiv:2310.11454](https://arxiv.org/abs/2310.11454)
- *SVFit: Parameter-Efficient Fine-Tuning of Large Pre-Trained Models Using Singular Values*. [arXiv:2409.05926](https://arxiv.org/abs/2409.05926)
- Lingam et al. (2024), *SVFT: Parameter-Efficient Fine-Tuning with Singular Vectors*, NeurIPS. [neurips.cc](https://neurips.cc/virtual/2024/poster/93272)
- *Transformer²: Self-adaptive LLMs*. [arXiv:2501.06252](https://arxiv.org/abs/2501.06252)
- M. Wardermann, J. Steil (2007), *Intrinsic plasticity for reservoir learning algorithms*, ESANN. [PDF](https://www.esann.org/sites/default/files/proceedings/legacy/es2007-98.pdf) (résumé)
- B. Schrauwen et al. (2008), plasticité intrinsèque pour réservoirs à `tanh` (non lu ; référence exacte à retrouver).
- *Federated Adaptation of Reservoirs via Intrinsic Plasticity*. [arXiv:2206.11087](https://arxiv.org/abs/2206.11087) (résumé)
- D. Di Sarli, C. Gallicchio, A. Micheli (2022), *On the effectiveness of Gated Echo State Networks for data exhibiting long-term dependencies*, Computer Science and Information Systems. [doi:10.2298/CSIS210218063D](https://zenodo.org/record/7040474)
- *Linear Reservoir: A Diagonalization-Based Optimization* (2026). [arXiv:2602.19802](https://arxiv.org/abs/2602.19802) ; *Efficient Time-Series Approximation with Linear Recurrent Neural Networks* (2018). [arXiv:1802.03308](https://arxiv.org/pdf/1802.03308) (résumés)
- *Recurrent Spectral Network (RSN)* (2022). [arXiv:2202.04497](https://arxiv.org/abs/2202.04497) (résumé)

## Parties 5 et 6 — non-normalité

- S. Ganguli, D. Huh, H. Sompolinsky (2008), *Memory traces in dynamical systems*, PNAS.
- L. N. Trefethen, M. Embree (2005), *Spectra and Pseudospectra: The Behavior of Nonnormal Matrices and Operators*, Princeton University Press.
- G. Kerg, K. Goyette, M. Puelma Touzel, G. Gidel, E. Vorontsov, Y. Bengio, G. Lajoie (2019), *Non-normal Recurrent Neural Network (nnRNN): learning long time dependencies while improving expressivity with transient dynamics*, NeurIPS. [arXiv:1905.12080](https://arxiv.org/abs/1905.12080) (résumé et page NeurIPS)
- A. E. Orhan, X. Pitkow (2020), *Improved memory in recurrent neural networks with sequential non-normal dynamics*, ICLR (référence à vérifier).
- *Euler State Networks: Non-dissipative Reservoir Computing*. [lien](https://astro.paperswithcode.com/paper/euler-state-networks) ; *Lindblad-Inspired Multi-Timescale Reservoir Computing* (2026). [arXiv:2608.04028](https://arxiv.org/pdf/2608.04028) (résumés)

## Partie 7 — coupler

- R. Pascanu, C. Gulcehre, K. Cho, Y. Bengio (2014), *How to Construct Deep Recurrent Neural Networks* (DT-RNN).
- J. Chung, C. Gulcehre, K. Cho, Y. Bengio (2015), *Gated Feedback Recurrent Neural Networks* (GF-RNN).
- J. Pathak et al. (2018), *Hybrid forecasting of chaotic processes: using machine learning in conjunction with a knowledge-based model*, Chaos.
- D. J. Gauthier et al. (2021), *Next generation reservoir computing* (NG-RC / NVAR), Nature Communications.
- S. Shen et al. (2021), *Reservoir Transformers*, ACL. [aclanthology 2021.acl-long.331](https://aclanthology.org/2021.acl-long.331)
- Autres voisins lus à travers leur résumé seulement (références complètes à vérifier) : LoRA-RC (2026), Inter-ESN, MCI-ESN, HBP-ESN, Deep Residual ESN, Product RC, full-FORCE, GRU-ESN, FreezeTST.

## Partie 8 — positionnement, données, bancs d'essai

- *A Little Rank Goes a Long Way: Random Scaffolds with LoRA Adapters Are All You Need* (LottaLoRA, 2026). [arXiv:2604.08749](https://arxiv.org/abs/2604.08749) (lu : sections 1 à 5 et annexe C)
- M. C. Mackey, L. Glass (1977), *Oscillation and chaos in physiological control systems*, Science.
- H. Jaeger, H. Haas (2004), *Harnessing Nonlinearity: Predicting Chaotic Systems and Saving Energy in Wireless Communication*, Science (égalisation de canal).
- A. A. M. Saleh (1981), *Frequency-Independent and Frequency-Dependent Nonlinear Models of TWT Amplifiers*, IEEE Trans. Communications (amplificateur).
- R. H. Clarke (1968), *A Statistical Theory of Mobile-Radio Reception*, Bell System Technical Journal (canal de Rayleigh).
- J. Kim, K. Konstantinou (2001), *Digital predistortion of wideband signals based on power amplifier model with memory*, Electronics Letters 37(23) ; D. R. Morgan et al. (2006), *A generalized memory polynomial model for digital predistortion of RF power amplifiers*, IEEE Trans. Signal Processing 54(10).
- *OpenDPD: An Open-Source End-to-End Learning & Benchmarking Framework for Wideband Power Amplifier Modeling and Digital Pre-Distortion*. [arXiv:2401.08318](https://arxiv.org/abs/2401.08318) (résumé)
- T. Wigren, J. Schoukens (2013), *Three free data sets for development and benchmarking in nonlinear system identification*, ECC (Silverbox).
- J. Schoukens, J. Suykens, L. Ljung (2009), *Wiener-Hammerstein benchmark*, SYSID.
- M. Schoukens, P. Mattsson, T. Wigren, J.-P. Noël (2016), *Cascaded tanks benchmark combining soft and hard nonlinearities*.
- G. Beintema, R. Tóth, M. Schoukens (2021), *Nonlinear state-space identification using deep encoder networks*, L4DC. [arXiv:2012.07697](https://arxiv.org/abs/2012.07697)
- G. Beintema, M. Schoukens, R. Tóth (2023), *Deep subspace encoders for nonlinear system identification*, Automatica. [arXiv:2210.14816](https://arxiv.org/abs/2210.14816) ; *Continuous-time identification of dynamic state-space models by deep subspace encoding*, ICLR. [arXiv:2204.09405](https://arxiv.org/abs/2204.09405)
- M. D. Champneys, G. I. Beintema, R. Tóth, M. Schoukens, T. J. Rogers (2024), *Baseline Results for Selected Nonlinear System Identification Benchmarks*. [arXiv:2405.10779](https://arxiv.org/abs/2405.10779)
- A. Bagnall et al. (2018), *The UEA multivariate time series classification archive, 2018*. [arXiv:1811.00075](https://arxiv.org/abs/1811.00075)
- H. Zhou et al. (2021), *Informer* (données ETTh1, [github.com/zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset), CC BY-ND 4.0).

## Partie 9 — la tâche peut-elle choisir sa structure ?

- Y.-L. Sung, V. Nair, C. Raffel (2021), *Training Neural Networks with Fixed Sparse Masks* (FISH mask), NeurIPS. [arXiv:2111.09839](https://ar5iv.arxiv.org/html/2111.09839)
- Y. Lee, A. S. Chen, F. Tajwar, A. Kumar, H. Yao, P. Liang, C. Finn (2022), *Surgical Fine-Tuning Improves Adaptation to Distribution Shifts*. [arXiv:2210.11466](https://ar5iv.labs.arxiv.org/html/2210.11466)
- M. S. Abdelfattah, A. Mehrotra, Ł. Dudziak, N. D. Lane (2021), *Zero-Cost Proxies for Lightweight NAS*, ICLR. [arXiv:2101.08134](https://alphaxiv.org/abs/2101.08134)
- S. Chatterjee (2020), *Coherent Gradients: An Approach to Understanding Generalization in Gradient Descent-based Optimization*, ICLR.
- G. Gur-Ari, D. A. Roberts, E. Dyer (2018), *Gradient Descent Happens in a Tiny Subspace*. [arXiv:1812.04754](https://alphaxiv.org/abs/1812.04754)
- C. Li, H. Farkhoor, R. Liu, J. Yosinski (2018), *Measuring the Intrinsic Dimension of Objective Landscapes*, ICLR. [arXiv:1804.08838](https://alphaxiv.org/abs/1804.08838)
