[Sommaire](../README.md#le-rapport) · [← Introduction](0_introduction.md) · [Partie 2 →](2_sculpte.md)

# Partie 1 — Mélanger un réservoir et un RNN : c'est freiner

![Le mélange est un freinage](../results/figures/schema_partie1_freine.png)

> **Ce qu'il faut retenir**
>
> - Le mélange `W = (1−α)·W₀ + α·ΔW` ne mélange rien : avec Adam, il revient exactement à apprendre α fois plus lentement.
> - La règle qui en sort, le **RNN freiné** (entrées à pleine vitesse, récurrence 10 fois plus lente), fait 2 à 2,6 fois moins d'erreur que le meilleur RNN à 100 neurones, et l'écart grandit avec la taille. Avec un α appris par matrice, le réseau retrouve cette règle tout seul.
> - C'est très proche d'une prescription connue de la théorie des grands réseaux (μP) et de la lignée APRL/BPDC des réservoirs : l'intérêt est d'y arriver par le mélange, pas la nouveauté.

L'idée la plus directe pour « mélanger » un réservoir et un RNN est d'écrire la matrice récurrente comme une moyenne :
une part fixe de hasard, une part apprise. On la teste, on comprend pourquoi elle se réduit à un freinage, et on en tire
une règle simple.

![Résultat principal](../results/figures/headline.png)

| Tâche | Réservoir classique | Meilleur RNN | RNN freiné | Mélange réglé | Réservoir retouché |
|---|---|---|---|---|---|
| Mackey-Glass (+15 pas) | 0.0018 | 0.000085 | **0.000032** | 0.000035 | 0.000045 |
| NARMA10 | 0.279 | 0.0173 | **0.0082** | 0.0088 | 0.0094 |
| Mémoire (délai 15) | 0.785 | 0.0027 | 0.0011 | **0.0010** | 0.0013 |

NMSE de test (plus bas = mieux), 100 neurones, moyenne sur 8 graines (5 pour la mémoire). « Meilleur RNN » : le meilleur
de trois learning rates pour chaque tâche. Tous les écarts au meilleur RNN sont significatifs (|t| > 4 sur les différences
appariées).

## 1.1 Le mélange fixe est un RNN ralenti

Avec Adam, mélanger `(1−α)·W₀ + α·ΔW` donne **les mêmes résultats** qu'un RNN dont tous les poids apprennent α fois moins
vite (à 2 % près sur Mackey-Glass et la mémoire, à 11 % près sur NARMA10). La raison tient en une ligne : le gradient de
`ΔW` est multiplié par α, Adam normalise l'échelle des gradients, donc le pas sur `ΔW` est inchangé, et le pas effectif sur
`W` est multiplié par α. Le mélange n'est donc pas une architecture à part : c'est un frein sur l'apprentissage.

![Mélange fixe contre RNN ralenti](../results/figures/alpha_vs_lr.png)

## 1.2 Freiner la récurrence bat le RNN, même bien réglé

La crainte naturelle : le RNN serait juste mal réglé. Il reçoit donc trois learning rates (standard, × 0,3, × 0,1) et on
compare au meilleur des trois, tâche par tâche. Écart apparié `log10(NMSE / meilleur RNN)` :

| | Mackey-Glass | NARMA10 | Mémoire |
|---|---|---|---|
| RNN freiné (récurrence 10 fois plus lente) | −0,41 ± 0,05 | **−0,32 ± 0,02** | −0,39 ± 0,06 |
| Mélange réglé (α appris) | −0,39 ± 0,08 | −0,29 ± 0,02 | **−0,41 ± 0,05** |
| Réservoir retouché | −0,25 ± 0,04 | −0,26 ± 0,03 | −0,31 ± 0,04 |
| Réservoir corrigé de rang 4 | −0,09 ± 0,08 | −0,16 ± 0,04 | −0,35 ± 0,05 |

Baisser le learning rate **global** du RNN ne suffit pas : sur Mackey-Glass, ça ne change rien ou ça le dégrade ; sur la
mémoire, ça le dégrade ; sur NARMA10, ça l'améliore (0,032 → 0,017) mais il reste deux fois derrière. Ce qui compte, c'est
de freiner **spécifiquement** la dynamique récurrente.

![Toutes les variantes](../results/figures/comparison.png)

## 1.3 Freiner aussi les entrées est une erreur

Si on ralentit tout (entrées **et** récurrence), ça marche sur Mackey-Glass et NARMA10, mais ça échoue sur la mémoire
(4 fois moins bien que le RNN) : pour se souvenir, le réseau doit apprendre à bien injecter l'entrée.

| | Mackey-Glass | NARMA10 | Mémoire |
|---|---|---|---|
| RNN | 0.000085 | 0.0316 | 0.0027 |
| RNN ralenti × 0,1 (entrées et récurrence) | 0.000029 | 0.0125 | 0.0105 |
| RNN freiné (récurrence seule × 0,1) | 0.000032 | **0.0082** | **0.0011** |

Le même effet se voit sur MNIST lu ligne par ligne (256 neurones, 3 époques, mêmes réglages pour tous) : 4,96 % d'erreur
pour le RNN freiné contre 7,01 % pour le RNN ralenti.

## 1.4 Le réseau trouve la règle tout seul

Avec le **mélange réglé** (`α_in` et `α_rec` appris par rétropropagation, départ à 0,1), le réseau :

- fait monter `α_in` selon le besoin de la tâche : **0,21** (Mackey-Glass), **0,66** (NARMA10), **0,90** (mémoire) ;
- garde `α_rec` bas sur toutes les tâches : **0,16 à 0,19**.

Il choisit donc d'apprendre ses entrées et de garder une dynamique proche du hasard. C'est le meilleur modèle sur la
mémoire et le deuxième sur NARMA10. Limite : le résultat dépend du point de départ ; en partant de α = 0,5, les α restent
plus hauts et les scores sont moins bons (mais toujours meilleurs que le RNN).

![Mélange réglé](../results/figures/learned_alpha.png)

## 1.5 Ce que le freinage change dans la matrice

Sur une graine par tâche, on compare le spectre du réservoir de départ à celui des matrices apprises :

| | Rayon spectral | Plus grande valeur singulière | Changement relatif de `W` |
|---|---|---|---|
| Réservoir de départ | 0,90 | 1,75 | 0 |
| RNN — Mackey-Glass / NARMA10 / mémoire | 1,47 / 1,30 / 0,81 | 2,38 / 2,31 / 2,51 | 0,35 / 0,48 / 0,38 |
| RNN freiné — mêmes tâches | 0,98 / 1,11 / 0,86 | 1,78 / 1,87 / 1,80 | 0,12 / 0,18 / 0,15 |

Le RNN classique pousse des valeurs propres hors du cercle unité sur les tâches de prédiction. Sur la mémoire, il fait
l'inverse : il **baisse** le rayon spectral mais **augmente** la plus grande amplification, un mécanisme connu de mémoire
transitoire des réseaux non normaux (Ganguli et al., 2008), que les parties 5 et 6 testent directement. Le RNN freiné modifie
2,5 à 3 fois moins la matrice et reste dans le régime du réservoir.

![Spectres](../results/figures/spectra.png)

La capacité mémoire (partie 4.3) dit la même chose de l'intérieur : à 100 neurones, le RNN freiné garde une mémoire plus
longue que le RNN (26 contre 22 entraîné sur la mémoire) et calcule davantage de non-linéaire sur NARMA10 (45 contre 27).

## 1.6 L'avantage grandit avec la taille

Écart au RNN de même taille, sur 5 graines :

| | N = 50 | N = 100 | N = 200 |
|---|---|---|---|
| Mackey-Glass, RNN freiné | ≈ 0 (+0,02) | −0,43 | **−1,20** (16 fois) |
| NARMA10, RNN freiné | ≈ 0 (−0,01) | −0,60 | **−0,93** (8,5 fois) |

Avec un petit réseau, aucun avantage ; avec 200 neurones, l'écart devient très grand, en partie parce que le RNN classique
se dégrade (NARMA10 : 0,034 à N = 100, 0,087 à N = 200) alors que la variante freinée continue de s'améliorer. Côté rayon
spectral du réservoir : avec ρ = 0,6, les variantes freinées battent nettement le RNN (1,7 à 3 fois) ; avec ρ = 1,2,
l'écart est grand sur NARMA10 (3 à 4 fois) mais faible sur Mackey-Glass (1,2 à 1,4 fois).

![Robustesse](../results/figures/robustness.png)

À 500 neurones (3 graines), le RNN se règle mal (une graine sur trois est 10 fois moins bonne au learning rate standard).
Contre le meilleur de quatre learning rates, le RNN freiné fait 6 fois mieux sur Mackey-Glass et 3 fois mieux sur NARMA10 ;
le mélange réglé 10 et 3 fois. Mais à cette taille, les modèles plus contraints des parties suivantes passent devant : le
réservoir corrigé de rang 4 fait 13 et 9 fois mieux que le RNN, le réservoir accordé 15 et 9 fois.

## 1.7 Deux autres façons de rester près du réservoir

- **Le réservoir retouché** (`W = W₀ + M ⊙ ΔW`) n'apprend que les connexions qui existent déjà dans le réservoir creux.
  Il bat le RNN de 1,8 à 2 fois à 100 neurones, et c'est le meilleur modèle sur MNIST ligne par ligne (4,01 % contre 4,41 %
  pour le RNN), mais un RNN qui part d'un réservoir **dense** fait aussi bien (4,05 %) : sur MNIST, son avantage ne vient
  pas clairement de la contrainte.
- **Le réservoir corrigé de rang r** (`W = W₀ + U·Vᵀ`) n'apprend que quelques directions. À 100 neurones, le rang 4 gagne
  surtout sur la mémoire (2,2 fois) ; à 500 neurones, il devient l'un des meilleurs modèles (au niveau de l'accordé sur
  NARMA10, 0,0101) ; à 2 000 neurones, c'est le meilleur sur Mackey-Glass (3,2e-9). Le rang 16 est moins bon que le rang 4
  partout : plus de liberté n'aide pas.

## 1.8 Bilan de la partie 1

**Où ça marche.** Sur les trois tâches principales, à 100 et 200 neurones, freiner la récurrence (RNN freiné, mélange
réglé) bat le RNN bien réglé de 2 à 16 fois, et l'avantage grandit avec la taille. Garder les entrées libres est
indispensable dès que la tâche demande de la mémoire.

**Les limites.**
- À 500 neurones, le freinage est dépassé par des modèles qui apprennent bien moins (rang 4, accordé), et l'ajouter à
  ces modèles **dégrade** toujours (jusqu'à 14 fois sur Mackey-Glass) : un modèle déjà contraint n'a pas besoin d'être
  ralenti en plus.
- Sur MNIST ligne par ligne, freiner ne paie pas en 3 époques (RNN freiné 4,96 % contre 4,41 % pour le RNN). Sur trois
  séries réelles (température d'un transformateur, température à Melbourne, taches solaires), aucun modèle entraîné ne fait
  mieux que le réservoir classique (partie 8.3).
- Le résultat sur α appris dépend du point de départ ; Lorenz et Hénon n'ont pas été mesurés pour ces modèles.

**Le lien avec la théorie.**
- *Adam et le mélange.* L'équivalence entre mélange et ralentissement est une conséquence directe de l'invariance d'Adam
  à l'échelle du gradient (section 1.1).
- *μP.* Dans la théorie des réseaux larges ([Yang et Hu, 2021](https://arxiv.org/abs/2011.14522) ;
  [Yang et al., 2022](https://arxiv.org/abs/2203.03466)), la paramétrisation μP prescrit, avec Adam, de garder le learning
  rate des poids d'entrée constant et de diviser celui des poids cachés par la largeur. Le RNN freiné, et le fait que son
  avantage grandisse avec N, ressemblent beaucoup à cette prescription : c'est l'explication la plus probable, et la
  principale limite à la nouveauté. Les validations publiées de μP portent sur des MLP, des CNN et des Transformers ; un test
  explicite sur des RNN issus d'un réservoir (le learning rate récurrent optimal varie-t-il comme 1/N ?) reste à faire.
- *Réservoirs aux poids internes lents.* Les méthodes Atiya-Parlos et la rétropropagation-décorrélation (BPDC, Steil) apprennent
  la sortie vite et les poids internes lentement ou peu ([Lukoševičius et Jaeger, 2009](https://www.ai.rug.nl/minds/uploads/2261_LukoseviciusJaeger09.pdf)) :
  le RNN freiné appartient à cette lignée.
- *Apprentissage paresseux et faible rang.* Une connectivité initiale de haut rang rend l'apprentissage plus « paresseux »
  ([Liu et al., ICLR 2024](https://arxiv.org/abs/2310.08513)) ; ici, freiner l'éloignement au réservoir **améliore** la
  performance. Les changements appris par un RNN initialisé au hasard sont naturellement de faible rang
  ([Schuessler et al., 2020](https://arxiv.org/abs/2006.11036)), ce qui explique qu'une correction de rang 4 perde peu.
  « Base aléatoire figée + correction de faible rang » est par ailleurs le principe de LottaLoRA (partie 8.4), et le
  réservoir retouché reprend l'idée de [Palangi et al. (2013)](https://arxiv.org/abs/1311.2987), qui n'apprennent que les
  connexions non nulles d'un réservoir creux.

---

[Sommaire](../README.md#le-rapport) · [← Introduction](0_introduction.md) · [Partie 2 →](2_sculpte.md)
