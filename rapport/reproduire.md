[Sommaire](../README.md#le-rapport)

# Reproduire

## Installation et vérification

```bash
pip install -e ".[dev]"          # numpy, scipy, matplotlib, pytest  (".[dev,gpu]" pour PyTorch)
pytest                           # quelques minutes ; les tests PyTorch sont ignorés sans PyTorch
```

Les expériences à 500 neurones et plus tournent sur GPU (Google Colab ou Kaggle) avec
[`notebooks/colab_gpu.ipynb`](../notebooks/colab_gpu.ipynb), qui contient les mesures encore en attente (voir
l'[annexe](annexe_pistes.md)). Les résultats déjà calculés sont dans `results/data/` : chaque script reprend ce qui existe et
ne calcule que ce qui manque. Tous les runs numpy sont gardés dans `results/data/cache.json` ; relancer un script déjà calculé
est instantané. Pour un essai rapide sans cache :
`HFRNN_STEPS=20 HFRNN_RESULTS_DIR=/tmp/essai python experiments/03_learned_alpha.py`.

## Figures sans calcul

```bash
python experiments/38_schemas.py          # tous les schémas pédagogiques (quelques secondes)
python experiments/42_figures_rapport.py  # bilan des familles, qui apprend quoi, budget égal, recherche équitable
python experiments/28_synthesis.py        # synthèses par tâche et front erreur/paramètres (partie 8)
python experiments/11_model_gallery.py    # galerie : ce que chaque modèle fait au réservoir (~1 min)
python experiments/20_pitch_figure.py     # figure d'ouverture de la partie 8
python experiments/01_esn_demo.py         # démo : un réservoir sur Fibonacci mod 10
```

## Expériences, partie par partie

Les scripts sont numérotés dans l'ordre où ils ont été écrits ; la liste les range selon le plan du rapport.

```bash
# Partie 1 — mélanger, freiner (CPU)
python experiments/02_comparison.py        # 1.1 à 1.3, 1.7       (~45 min sans cache)
python experiments/03_learned_alpha.py     # 1.4                  (~15 min)
python experiments/09_rotation.py          # 1.5 (spectres) et 3.1 (~20 min)
python experiments/04_robustness.py        # 1.6                  (~1 h)
# Partie 2 — réservoir sculpté
python experiments/08_supermask.py         # 2.1 à 2.3 (CPU, ~30 min)
# Partie 3 — réservoir tourné
python experiments/09_rotation.py          # 3.1 (CPU)
python experiments/36_decompositions.py --n 100 --seeds 3   # 3.2, 5.1, 6.1 (cinq tâches, 100 neurones)
# Partie 4 — réservoir accordé
python experiments/05_gpu_regression.py --n 500 --seeds 3   # 4.1 et combinaisons de 4.4 (GPU)
python experiments/37_budget_and_inputs.py # 4.2 (qui apprend quoi) et 4.9 (budget égal) (GPU)
python experiments/40_truncated_tuned.py   # 4.2, accordé tronqué (GPU)
python experiments/18_spectrum_controls.py # 4.2, contrôles du spectre (GPU)
python experiments/24_memory_capacity.py   # 4.3, capacité mémoire (CPU, ~10 min)
python experiments/13_exploration.py       # 4.4, combinaisons à 100 neurones (CPU, ~35 min)
python experiments/15_improvements.py --n 500   # 4.4, treize variantes (GPU) ; 17_tuned_gallery.py : galerie
python experiments/21_gated.py             # 4.5 et 4.6, portes et récurrence linéaire (GPU)
python experiments/22_gated_gallery.py     # 4.5, galerie ; 32_linear_gallery.py : 4.6, galerie
python experiments/16_scaling.py           # 4.7, de 100 à 2 000 neurones (GPU)
python experiments/19_fast_scaling.py      # 4.7, accordé rapide jusqu'à 16 000 neurones (GPU)
python experiments/25_multitask.py         # 4.8 (CPU à 100 neurones ; --n 500 sur GPU)
# Parties 5 et 6 — réservoir cisaillé, réservoir nilpotent
python experiments/36_decompositions.py --n 100 --seeds 3   # 5.1 et 6.1 ; --n 500 : 5.3 et 6.2 (pleins et en bande)
python experiments/36b_decompositions_figure.py             # figure de 5.1
# Partie 7 — couplages
python experiments/35_coupled_forms.py     # 7.2 (--n 100, CPU ~30 min) et 7.3 (--n 500, GPU)
python experiments/39_combinations.py      # 7.4 (GPU)
python experiments/41_readout_features.py  # 7.5 (GPU)
# Partie 8 — synthèse, modèles existants, données non synthétiques
python experiments/12_comparisons.py       # 8.2, premier protocole et portes à 4 000 pas (GPU ; --report sans GPU)
python experiments/26_fair_search.py       # 8.2, même budget de recherche, tailles comprises (GPU ; --quick : test CPU)
python experiments/06_gpu_smnist.py        # 8.3, MNIST ligne par ligne (GPU)
python experiments/23_mnist_search.py      # 8.3, MNIST avec recherche (GPU)
python experiments/27_pixel_mnist.py       # 8.3 et 4.6, MNIST pixel par pixel (GPU)
python experiments/31_uea.py               # 8.3, archive UEA (GPU ; pip install aeon)
python experiments/14_real_data.py         # 8.3, séries réelles (CPU, ~15 min)
python experiments/29_task_map.py          # 8.3, carte des tâches (GPU)
python experiments/33_amplifier_control.py # 8.3, amplificateur, premier contrôle (GPU)
python experiments/34_amplifier_deep_dive.py   # 8.3, amplificateur, sur-apprentissage et polynôme à mémoire (GPU)
python experiments/30_nonlinear_benchmarks.py  # 8.3, systèmes réels (GPU ; pip install nonlinear-benchmarks ;
                                               #      --extra : parties 6 et 7 ; --extra2 : sans g et cisaillé)
# Partie 5.4 à 5.10 — confirmation du cisaillé en bande (un processus par graine, plusieurs à la fois sur un GPU)
python experiments/43_shear_study.py curve --seeds 0     # 5.4 et 5.5 : courbe en k, contrôles (positions au hasard, dans W₀)
python experiments/43_shear_study.py budget --seeds 0    # 5.7 : budget égal, avec les résultats de budget_inputs.json
python experiments/43_shear_study.py gated --seeds 0     # 5.10 : cisaillé à portes à 200 neurones
python experiments/43_shear_study.py fresh --seeds 0     # 5.6 : séries régénérées, réglages fixés d'avance
python experiments/43_shear_study.py real --seeds 0      # 5.8 : systèmes réels
python experiments/43_shear_study.py freshreal --seeds 0 # 5.6 : EMPS
python experiments/43_shear_study.py curve --report      # fusionne les fichiers par graine, tableaux et figure
python experiments/26_fair_search.py --tasks "Lorenz (+10)" --trials 8 --tag _lorenz --models "cisaillé"   # 5.9 et 8.2
# Partie 9 — la tâche peut-elle choisir sa structure ? (CPU, sauf mention)
python experiments/36_decompositions.py --n 100 --seeds 3 --tag _p9a --only "en bande" "sans g"   # vérité terrain (~15 min)
python experiments/44_structure_signals.py   # 9.3 : signaux du gradient au départ, évaluation (~30 min)
python experiments/45_signaux_dynamiques.py --seed 0   # 9.4 : signaux en cours d'entraînement (une graine ; puis --report)
python experiments/46_sous_espace_appris.py --seed 0   # 9.5 : sous-espaces tirés du gradient (une graine ; puis --report)
python experiments/47_protocole_lie.py --seed 0   # 9.5 : PCA du gradient tangent M⁻¹G, blocs et dégel (une graine ; puis --report)
python experiments/48_enseignants.py --seed 0     # 9.6 : tâches à structure plantée (une graine ; puis --report)
python experiments/49_preenregistre.py --phase calibrage --amps 0.2   # 9.7 : vérification sur des graines neuves, voir l'en-tête du script
```

## Correspondance des noms

Quelques modèles portent dans les fichiers de résultats le nom technique de leur décomposition :

| Nom dans le rapport | Nom dans `results/data/` |
|---|---|
| Réservoir cisaillé (partie 5) | « Iwasawa : cisaillement (N) » (`form="iwasawa_N"`) |
| Réservoir cisaillé + amplitudes | « Iwasawa : amplitudes et cisaillement (A et N) » |
| Réservoir nilpotent (partie 6) | « Jordan-Chevalley + partie nilpotente apprise » (`form="jordan_SN"`) |
| Accordé sur les valeurs propres | « accordé, valeurs propres (Jordan-Chevalley, partie semi-simple) » |
| Réservoir à biais et entrées appris | « contrôle : biais + entrées W_in, sans g » (`learn_g=False`) |
| Relais aléatoire, relais appris, duos croisés | « forme 3 », « forme 4 », « formes 1 et 2 » |

## Organisation du code

```
partially-frozen-rnn/
├── src/hfrnn/
│   ├── core.py            # réservoir, ridge, rétropropagation à travers le temps, Adam
│   ├── tasks.py           # Mackey-Glass, NARMA10, mémoire, Fibonacci (démo), séries réelles
│   ├── task_map.py        # les douze tâches de la carte des tâches et la régression linéaire de référence
│   ├── params.py          # les modèles des parties 1 à 3 (numpy) et l'entraînement
│   ├── runner.py          # grilles d'expériences en parallèle, cache, comparaisons appariées
│   ├── torch_models.py    # version PyTorch (GPU)
│   ├── torch_baselines.py # GRU, LSTM, LRU, et l'accordé linéaire
│   ├── torch_tuned.py     # le réservoir accordé et ses extensions (partie 4)
│   ├── torch_decomp.py    # réservoirs cisaillé et nilpotent, pleins ou en bande (parties 5 et 6)
│   ├── torch_shear.py     # réservoir cisaillé à portes : blocs d'un GRU figés, cisaillés ou appris (partie 5)
│   ├── structure_signals.py  # espaces tangents des structures, gradients au départ, signaux (partie 9)
│   └── torch_coupled.py   # les quatre couplages et le readout carré (partie 7)
├── experiments/           # un script par expérience
├── notebooks/             # le carnet Colab des mesures en attente
├── tests/
├── results/               # figures et données
└── fausses_pistes/        # essais abandonnés, et pourquoi
```

Le dossier [`fausses_pistes/`](../fausses_pistes/README.md) garde les essais qui ont mené à des conclusions trompeuses.
