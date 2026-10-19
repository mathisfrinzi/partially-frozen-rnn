[Sommaire](../README.md#le-rapport) · [← Partie 8](8_synthese.md) · [Annexe →](annexe_pistes.md)

# Partie 9 — La tâche peut-elle choisir sa structure ?

> **Ce qu'il faut retenir**
>
> - Les parties 1 à 8 fixent à la main ce que la récurrence apprend (tourner, accorder, cisailler), et la meilleure structure change d'une tâche à l'autre. Cette partie cherche un signal, mesuré sur les données, qui dise quelle structure apprendre : le gradient de la perte, projeté sur l'espace tangent de chaque structure.
> - **Le gradient ne le dit pas.** Au départ, ses signaux classent les structures comme leur simple taille, et le choix naïf (« la structure la meilleure en moyenne sur les autres tâches ») fait nettement mieux. Sonder plus tard dans l'entraînement n'aide pas : l'information propre à la tâche qui apparaît à 100 pas ne donne pas de bons choix, et l'enjeu du choix disparaît à mesure que le RNN apprend.
> - **Il ne désigne pas non plus de sous-espace.** Des directions tirées du gradient, y compris les directions propres de la covariance du tangent `M⁻¹G` (une décomposition de Lie apprise sur la tâche), ne font pas mieux que des directions aléatoires de même dimension, et l'ordre de dégel par valeurs propres ne compte pas. Cette covariance est presque de rang un.
> - **Même quand la structure est plantée dans la tâche, et atteignable, ni le gradient ni l'entraînement ne la retrouvent.** Sur des enseignants construits pour chaque structure, aucun signal ne retrouve la structure plantée, y compris sur des graines neuves. Plus surprenant : à budget égal et avec arrêt anticipé, la structure plantée ne gagne presque jamais sa propre tâche ; ce sont les bandes, quelques centaines de coefficients près de la diagonale, qui gagnent.
> - **En une phrase : la facilité d'apprentissage bat la géométrie de la cible, même quand la solution exacte est dans la famille.** La performance dépend du nombre de coefficients appris dans la récurrence et de la facilité à les apprendre, pas de leur emplacement ni de la géométrie de la cible. Deux indices prometteurs (le signal sur bruit du gradient, les premières directions du gradient sur Lorenz et Hénon) ont été vérifiés sur des graines neuves : tous deux sont tombés.

## 9.1 La question

Chaque famille des parties 1 à 7 gagne sur ses tâches : le tourné sur Hénon, le cisaillé en bande sur Mackey-Glass, NARMA10 et
la mémoire, le RNN complet sur Lorenz (à 100 neurones). Rien ne dit à l'avance laquelle choisir. L'idée de cette partie est de
laisser l'entraînement le décider : partir d'un réservoir et ne dégeler que les pièces de la récurrence qui « veulent »
apprendre, ou partir d'un RNN et geler les autres. Il faut pour cela une grandeur, mesurée sur les données, qui dise vers
quelle pièce aller ; et que cette grandeur reflète une tendance de la tâche, pas le bruit d'un lot.

**Le cadre : la décomposition de l'algèbre de Lie.** Les parties 4 à 6 apprennent chacune un facteur d'une décomposition du
groupe des matrices inversibles (K·A·K, K·A·N, Jordan-Chevalley). Un gradient ne vit pas dans le groupe mais dans son algèbre
de Lie, `gl(n)`, l'espace de toutes les matrices N×N. La décomposition d'Iwasawa y prend sa forme infinitésimale :
`gl(n) = so(n) ⊕ a ⊕ n`, toute matrice s'écrivant de façon unique comme une partie antisymétrique (rotation), une partie
diagonale (amplitudes) et une partie strictement triangulaire (cisaillement). Chaque structure apprend, au premier ordre, un
sous-espace `T_s` de `gl(n)` : son **espace tangent** en `W₀`.

| Structure | Espace tangent en `W₀` (partie récurrente) | Dimension à 100 neurones |
|---|---|---|
| Accordé | `U·diag(δ)·Vᵀ` : les amplitudes singulières | 100 |
| Tourné | `ΩW₀ − W₀Ω`, `Ω` antisymétrique : la rotation | 4 950 |
| Cisaillé | `Q·T`, `T` strictement triangulaire : le facteur N | 4 950 |
| Cisaillé en bande k | idem, `T` limitée aux k sur-diagonales | k·N environ |
| Nilpotent | partie semi-simple (module et phase des valeurs propres) et triangle `T` dans une base orthogonale tirée au hasard | 5 050 |
| RNN | tout `gl(n)` | 10 000 |
| Biais et entrées seuls | rien dans la récurrence | 0 |

Toutes les structures apprennent aussi `W_in` et les biais (200 dimensions de plus). La décomposition `so(n) ⊕ a ⊕ n` est
directe mais **pas orthogonale** (une matrice antisymétrique et une matrice triangulaire ne sont pas orthogonales au sens de
Frobenius). Les « énergies » des trois parties d'un gradient ne s'additionnent donc pas ; chaque structure est traitée
séparément, par la projection orthogonale `P_s` sur son espace tangent. Les bases orthonormées des `T_s` sont obtenues à partir
de la jacobienne `∂W/∂θ_s` (autograd sur le modèle lui-même, formule exacte pour le tourné), et vérifiées : orthonormalité,
stabilité de la projection, dimension égale au nombre de paramètres. Le nilpotent fait exception (5 050 pour 5 150
paramètres) : sa partie semi-simple a deux paramètres par valeur propre, soit 200, mais seulement 100 degrés de liberté réels.
La phase d'une valeur propre réelle ne change rien au premier ordre (sa dérivée, `Re(iλ)`, est nulle), et les deux valeurs
propres d'une paire complexe conjuguée ne comptent que par la somme de leurs perturbations.

## 9.2 La vérité terrain

Treize structures (les neuf de la partie 5 à 100 neurones, les bandes k = 1, 3 et 10, le nilpotent en bande et le réservoir
à biais et entrées appris), cinq tâches, 3 graines, 1 500 pas ; on retient le log10 de la NMSE de test.

| log10 NMSE de test | Mackey-Glass | NARMA10 | Mémoire | Lorenz | Hénon |
|---|---|---|---|---|---|
| RNN | −4,10 | −1,49 | −2,55 | **−3,89** | −1,17 |
| Accordé | −4,05 | −1,63 | −2,54 | −3,21 | −1,38 |
| Tourné | −4,17 | −1,64 | −1,57 | −3,45 | **−1,82** |
| Valeurs propres (Jordan-Chevalley, semi-simple) | −4,11 | −1,51 | −2,13 | −2,96 | −1,38 |
| Nilpotent | −4,36 | −1,61 | −2,63 | −3,84 | −1,04 |
| Iwasawa, amplitudes A | −3,97 | −1,58 | −2,49 | −3,10 | −1,44 |
| Cisaillé plein | −4,13 | −1,70 | −2,60 | −3,83 | −1,68 |
| Cisaillé A et N | −4,19 | −1,67 | −2,62 | −3,86 | −1,24 |
| Bande k = 1 | −4,21 | −1,76 | −2,71 | −3,44 | −1,44 |
| Bande k = 3 | −4,40 | −1,89 | −2,80 | −3,52 | −1,36 |
| Bande k = 10 | **−4,47** | **−1,98** | **−2,84** | −3,81 | −1,74 |
| Nilpotent en bande, k = 3 | −4,29 | −1,83 | −2,76 | −3,62 | −1,45 |
| Biais et entrées seuls | −3,89 | −1,07 | −1,46 | −3,53 | −1,48 |

La meilleure structure est la bande k = 10 sur trois tâches, le RNN sur Lorenz, le tourné sur Hénon. L'écart entre la
meilleure et la médiane est de 0,24 à 0,37 en log10 : les structures diffèrent nettement, mais pas énormément.

## 9.3 Ce que le gradient dit au départ

Au réservoir de départ (biais nuls, `W_in` et `W₀` tirés), une lecture ridge est ajustée sur les états d'entraînement ; la
perte est celle de l'entraînement (erreur quadratique sur des fenêtres de 60 pas). On calcule le gradient
`G_i = ∂L_i/∂(W, W_in, b)` sur 128 fenêtres d'entraînement (75 % premiers de la partie entraînement) et 128 de validation
(25 % suivants), et on le projette sur chaque espace tangent.

| Signal | Définition | Ce qu'il teste |
|---|---|---|
| Part captée | `‖P_s(ḡ)‖² / ‖ḡ‖²` | quelle part de la descente de plus forte pente la structure peut suivre |
| Part par paramètre | idem, divisée par la dimension | la même chose, sans l'avantage de la taille |
| Signal sur bruit | `‖P_s(ḡ)‖²` divisé par la variance de la moyenne sur les fenêtres | une direction stable d'une fenêtre à l'autre s'apprend, le bruit se sur-apprend |
| Accord entraînement-validation | cosinus entre `P_s(ḡ)` sur l'entraînement et sur la validation | ce que la structure apprendra se généralise-t-il ? |
| Pas de Newton | `ḡᵀ(F_s + λI)⁻¹ḡ`, `F_s` Fisher empirique dans la structure | la baisse prédite en tenant compte de la courbure |
| Directions principales | part des 10 premières directions principales des gradients dans `T_s` | la structure contient-elle les directions que la perte explore ? |
| Mouvement naturel | `‖P_s(ΔW)‖² / ‖ΔW‖²`, `ΔW` = RNN entraîné − départ | où va un RNN qui apprend tout (coûte un entraînement) |
| Entraînement pilote | erreur de validation après 100 pas de la structure | référence chère |
| Taille | dimension de la structure | référence naïve |
| Choix naïf | classement moyen de la structure sur les quatre autres tâches | référence |

**Évaluation.** Pour chaque tâche et chaque graine, la corrélation de rang (Spearman) entre le signal et le gain mesuré, sur
les treize structures ; puis la structure choisie par tâche (rang moyen sur les graines) et son **regret**, l'écart en log10
NMSE à la meilleure. Enfin, pour isoler l'information **propre à la tâche** : signal et gain sont doublement centrés (on
retire la moyenne de chaque structure et celle de chaque tâche) puis corrélés ; un signal qui ne mesure que la taille ou la
difficulté générale d'une tâche y vaut zéro. Le p est obtenu en permutant les structures à l'intérieur de chaque tâche (2 000
permutations).

| Signal | Corrélation de rang | Bons choix | Regret moyen | Information propre à la tâche (r ; p) |
|---|---|---|---|---|
| Part captée | +0,23 ± 0,08 | 1 / 5 | 0,36 | +0,14 ; 0,37 |
| Part par paramètre | −0,26 ± 0,05 | 0 / 5 | 0,71 | +0,14 ; 0,31 |
| **Signal sur bruit** | +0,15 ± 0,08 | 1 / 5 | 0,27 | **+0,34 ; 0,02** |
| Accord entraînement-validation | +0,11 ± 0,10 | 0 / 5 | 0,38 | −0,03 ; 0,87 |
| Pas de Newton | +0,21 ± 0,08 | 1 / 5 | 0,36 | −0,25 ; 0,09 |
| Directions principales | +0,24 ± 0,08 | 1 / 5 | 0,36 | −0,15 ; 0,31 |
| Mouvement naturel | +0,19 ± 0,07 | 1 / 5 | 0,36 | +0,25 ; 0,06 |
| Entraînement pilote (100 pas) | +0,18 ± 0,09 | 1 / 5 | 0,25 | +0,07 ; 0,69 |
| Taille | +0,20 ± 0,07 | 1 / 5 | 0,36 | — |
| **Choix naïf** | **+0,45 ± 0,09** | **3 / 5** | **0,03** | — |
| Hasard | 0 | | 0,35 | |

![Signaux contre gain mesuré](../results/figures/structure_signals.png)

- **Aucun signal du gradient ne fait mieux que la taille.** Part captée, pas de Newton, directions principales et mouvement
  naturel ont la même corrélation que la dimension de la structure (0,19 à 0,24) et choisissent le RNN sur les cinq tâches :
  bon choix sur Lorenz, regret de 0,3 à 0,6 ailleurs. Diviser par la dimension inverse le biais sans rien corriger : la part
  par paramètre choisit partout le réservoir à biais et entrées appris, la pire structure sur trois tâches. La raison est
  simple : pour un gradient sans direction privilégiée, la part captée vaut en moyenne dim(T_s)/D, D étant la dimension de
  l'espace complet. Un signal qui ne bat pas ce rapport ne dit rien de la tâche.
- **L'entraînement pilote de 100 pas ne fait pas mieux** (0,18) : il ne trouve que le tourné sur Hénon. Les 100 premiers pas ne
  disent pas qui sera devant après 1 500.
- **Le choix naïf gagne largement** : la bande k = 10 est la meilleure ou presque partout (regret de 0,07 sur Lorenz, 0,08 sur
  Hénon). À cette taille, savoir quelle structure marche en général vaut plus que tout ce que le gradient dit de la tâche.
- **Un indice**, le signal sur bruit : une fois retirés les effets de la structure et de la tâche, il reste lié au gain
  (r = 0,34, p = 0,02). Avec huit signaux testés, la correction de Bonferroni demande p < 0,006 : ce n'est pas établi. Une
  direction de gradient cohérente d'un exemple à l'autre est celle qui généralise (cohérence des gradients, Chatterjee,
  2020) ; c'est l'indice suivi dans la suite de la partie (9.4, 9.6, 9.7).

![Signal sur bruit contre erreur après entraînement](../results/figures/structure_signals_scatter.png)

**Ce que le gradient voit** (moyennes sur les tâches et les graines) :

| Structure | Dimension | Part captée | Signal sur bruit | Accord entr.-valid. | Mouvement naturel |
|---|---|---|---|---|---|
| RNN | 10 200 | 1,00 | 12,2 | 0,19 | 1,00 |
| Nilpotent | 5 250 | 0,59 | 11,3 | 0,17 | 0,59 |
| Cisaillé plein | 5 150 | 0,51 | 10,7 | 0,14 | 0,59 |
| Tourné | 5 150 | 0,45 | 10,8 | 0,09 | 0,50 |
| Bande k = 10 | 1 145 | 0,26 | 8,2 | 0,08 | 0,22 |
| Bande k = 3 | 494 | 0,21 | 6,9 | 0,04 | 0,16 |
| Accordé | 300 | 0,19 | 6,5 | 0,03 | 0,16 |
| Biais et entrées seuls | 200 | 0,19 | 6,2 | 0,01 | 0,14 |

- **Les biais et `W_in` portent 19 % du gradient de départ** à eux seuls. Les structures à une centaine de paramètres
  récurrents (accordé, Iwasawa A, valeurs propres, bande k = 1) n'y ajoutent presque rien : au départ, le gradient de la
  récurrence tombe presque entièrement hors de leurs espaces tangents. Pourtant ces structures battent nettement le réservoir
  à biais et entrées appris sur Mackey-Glass, NARMA10 et la mémoire : leur gain se construit pendant l'entraînement, pas au
  premier pas.
- **Le gradient moyen est peu reproductible** : l'accord entre entraînement et validation ne dépasse pas 0,19, et le signal sur
  bruit tombe vers 1 sur NARMA10 et Hénon (contre 8 à 30 sur Mackey-Glass et Lorenz), c'est-à-dire un gradient moyen du même
  ordre que son incertitude.
- **Le mouvement naturel suit la part captée** : un RNN entraîné se déplace dans chaque structure à peu près en proportion de
  ce que le gradient de départ y met. Il ne dit pas mieux qu'elle laquelle choisir.

**Travaux voisins.** Choisir quoi entraîner à partir de gradients pris au départ est un domaine actif, toujours sur des
paramètres individuels ou des couches, jamais sur des sous-espaces structurés d'une matrice récurrente :
- le **FISH mask** ([Sung et al., NeurIPS 2021](https://ar5iv.arxiv.org/html/2111.09839)) garde les k paramètres de plus
  grande information de Fisher (diagonale, empirique, calculée une fois avant l'entraînement) et égale le réglage fin complet
  sur GLUE avec 0,5 % des paramètres ;
- le **surgical fine-tuning** ([Lee et al., 2022](https://ar5iv.labs.arxiv.org/html/2210.11466)) choisit les couches à
  régler selon le type de décalage des données ; son critère automatique par norme relative du gradient bat le réglage fin
  complet mais pas le meilleur choix manuel ;
- les **zero-cost proxies** de la recherche d'architecture ([Abdelfattah et al., ICLR 2021](https://alphaxiv.org/abs/2101.08134))
  prédisent le classement final d'architectures à partir d'un seul lot, avec une corrélation de rang jusqu'à 0,82 sur
  NAS-Bench-201. Ici, aucun signal ne dépasse 0,24, sous le choix naïf.

Ces trois travaux n'ont été lus qu'à travers leur résumé ou des extraits.

## 9.4 Sonder plus tard dans l'entraînement

Si le gain d'une structure se construit pendant l'entraînement, le signal se construit peut-être aussi. Le RNN s'entraîne t
pas (t = 0, 30, 100, 300, 1 000) ; à chaque t, on repose la question depuis l'état atteint `(W_t, W_in,t, b_t)` : quelle
structure apprendre pour la suite ? Cinq structures (RNN, accordé, tourné, cisaillé en bande k = 10, biais et entrées seuls)
sont redécomposées sur `W_t`, leurs biais partent de `b_t`, et les six signaux du gradient sont mesurés en `W_t`, ainsi que
l'entraînement pilote (100 pas depuis `W_t`). Le gain réel est le log10 NMSE de test après 500 pas de chaque structure depuis
`W_t`. Cinq tâches, 3 graines.

**La meilleure structure change avec t, et l'enjeu du choix fond.**

| t | Mackey-Glass | NARMA10 | Mémoire | Lorenz | Hénon | Écart moyen entre la meilleure et la pire des trois familles |
|---|---|---|---|---|---|---|
| 0 | cisaillé | cisaillé | cisaillé | cisaillé | tourné | 0,56 |
| 30 | tourné | cisaillé | cisaillé | cisaillé | tourné | 0,56 |
| 100 | tourné | cisaillé | cisaillé | cisaillé | tourné | 0,52 |
| 300 | tourné | tourné | accordé | tourné | tourné | 0,21 |
| 1 000 | cisaillé | cisaillé | tourné | cisaillé | tourné | 0,11 |

Au départ, le cisaillé gagne partout sauf sur Hénon, comme en 9.2 ; parmi les cinq structures, le RNN gagne Lorenz jusqu'à
t = 100. Après 300 pas de RNN, le classement se mélange. Après 1 000 pas, les trois familles sont à moins de 0,12 les unes des
autres sur quatre tâches (0,29 sur Hénon) : une fois le RNN bien avancé, le choix ne compte presque plus.

**Choix entre les trois familles** (bons choix sur 5 tâches / regret moyen en log10) :

| Signal | t = 0 | t = 30 | t = 100 | t = 300 | t = 1 000 |
|---|---|---|---|---|---|
| Part captée | 1 / 0,40 | 2 / 0,32 | 3 / 0,02 | 4 / 0,02 | 2 / 0,03 |
| Part par paramètre | 0 / 0,45 | 1 / 0,42 | 1 / 0,15 | 1 / 0,19 | 0 / 0,09 |
| Signal sur bruit | 2 / 0,22 | 1 / 0,27 | 2 / 0,12 | 1 / 0,14 | 1 / 0,09 |
| Accord entraînement-validation | 3 / 0,17 | 4 / 0,04 | 2 / 0,07 | 1 / 0,14 | 0 / 0,08 |
| Pas de Newton | 1 / 0,40 | 2 / 0,32 | 2 / 0,19 | 4 / 0,00 | 1 / 0,04 |
| Directions principales | 1 / 0,40 | 2 / 0,32 | 2 / 0,19 | 4 / 0,02 | 2 / 0,03 |
| Entraînement pilote (100 pas) | 3 / 0,14 | 4 / 0,06 | 1 / 0,26 | 2 / 0,07 | 2 / 0,06 |
| Taille | 1 / 0,40 | 2 / 0,32 | 2 / 0,19 | 4 / 0,02 | 2 / 0,03 |
| Hasard | 1,7 / 0,30 | 1,7 / 0,28 | 1,7 / 0,22 | 1,7 / 0,10 | 1,7 / 0,06 |

![Regret du choix selon t, et signal sur bruit du cisaillé selon t](../results/figures/signaux_dynamiques.png)

- **Aucun signal ne choisit durablement mieux que le hasard.** Les regrets baissent avec t pour tous les signaux, hasard
  compris (de 0,30 à 0,06) : c'est l'écart entre familles qui fond, pas le choix qui s'améliore.
- **Les signaux qui suivent la taille** choisissent surtout le tourné, le plus grand des trois (5 150 dimensions, contre 1 145
  pour la bande et 300 pour l'accordé) ; leurs 4 bons choix sur 5 à t = 300 viennent de ce que le tourné y gagne quatre tâches.
- **L'accord entraînement-validation et l'entraînement pilote font mieux au début** (3 et 4 bons choix sur 5 à t = 0 et 30),
  puis se perdent. Le pilote est le seul des deux qui coûte un entraînement.

**Information propre à la tâche** (signal et gain doublement centrés, parmi les cinq structures ; r, p par permutation) :

| Signal | t = 0 | t = 30 | t = 100 | t = 300 | t = 1 000 |
|---|---|---|---|---|---|
| Part captée | −0,19 ; 0,47 | +0,36 ; 0,14 | **+0,70 ; 0,003** | +0,29 ; 0,25 | +0,38 ; 0,18 |
| Signal sur bruit | −0,11 ; 0,67 | +0,46 ; 0,08 | **+0,78 ; < 0,001** | −0,26 ; 0,24 | −0,53 ; 0,07 |
| Accord entraînement-validation | +0,03 ; 0,92 | +0,25 ; 0,50 | **+0,74 ; 0,015** | −0,28 ; 0,27 | −0,15 ; 0,33 |
| Pas de Newton | +0,03 ; 0,93 | +0,35 ; 0,11 | **+0,62 ; 0,002** | +0,34 ; 0,14 | −0,38 ; 0,05 |
| Directions principales | +0,06 ; 0,82 | +0,28 ; 0,31 | −0,02 ; 0,95 | +0,45 ; 0,06 | +0,51 ; 0,08 |
| Entraînement pilote | +0,36 ; 0,18 | **+0,58 ; 0,02** | +0,56 ; 0,05 | +0,32 ; 0,16 | +0,01 ; 0,97 |

- **À t = 100, quatre signaux du gradient portent une information propre à la tâche** (r = 0,62 à 0,78), et seulement à ce
  moment-là. Avec 35 tests, la correction de Bonferroni demande p < 0,0014 : le signal sur bruit passe (r = 0,78, p < 0,001),
  ainsi que la part par paramètre (même r que la part captée, la division par la dimension disparaissant au double centrage).
- **Mais elle ne donne pas de bons choix** : à t = 100, le signal sur bruit ne trouve la meilleure des trois familles que sur
  2 tâches sur 5. En décomposant la corrélation, **64 % viennent des deux structures qui ne sont pas des familles** (biais et
  entrées seuls, 41 % ; RNN, 23 %). Restreinte aux trois familles, elle tombe à 0,62 (p = 0,045, non significatif après
  correction), et même en retirant à chaque structure sa moyenne sur les autres tâches avant de choisir, le signal ne trouve
  la meilleure famille qu'une fois sur 5. Ce qu'il voit, c'est quand le réservoir à biais et entrées seuls ou le RNN feront
  mieux ou moins bien que d'habitude, pas laquelle des trois familles gagne.
- Au départ (t = 0), le signal sur bruit ne retrouve pas l'indice de 9.3 : r = −0,11, contre 0,34 avec treize structures et
  1 500 pas.

## 9.5 Laisser le gradient désigner le sous-espace

Si le gradient ne choisit pas entre des structures fixées d'avance, il dit peut-être directement quelles directions de
`gl(n)` apprendre. Pour une dimension r (100, 300, 1 000 ; 1 000 = autant de coefficients que la bande k = 10), on apprend
`W = W₀ + Σ θ_j V_j` dans un sous-espace tiré du gradient au départ, avec les biais et `W_in` comme toujours.

**Des sous-espaces libres.**

| Sous-espace | Définition |
|---|---|
| Directions principales | les r premières directions principales (non centrées) des gradients de 1 280 fenêtres d'entraînement |
| Directions à accord | parmi elles, les r dont la composante moyenne est grande et de même signe sur l'entraînement et sur 256 fenêtres de validation |
| Positions du gradient | les r coefficients de W au plus grand gradient moyen (l'équivalent du FISH mask) |
| Positions à accord | les r coefficients au plus grand produit des gradients moyens d'entraînement et de validation |
| Contrôle : directions aléatoires | sous-espace orthonormé tiré au hasard (la « dimension intrinsèque » de Li et al., 2018) |
| Contrôle : positions au hasard | r coefficients tirés au hasard (le contrôle « appris dans W₀ » de la partie 5) |

Écart au contrôle aléatoire de même type et de même dimension (log10 NMSE, négatif = mieux ; moyenne sur les cinq tâches,
entre crochets le nombre de tâches où la méthode gagne ; 3 graines) :

| Méthode | r = 100 | r = 300 | r = 1 000 |
|---|---|---|---|
| Directions principales | −0,01 [2/5] | +0,02 [2/5] | +0,02 [2/5] |
| Directions à accord | −0,03 [3/5] | −0,08 [2/5] | −0,07 [3/5] |
| Positions du gradient | +0,17 [1/5] | +0,13 [0/5] | +0,13 [2/5] |
| Positions à accord | +0,16 [0/5] | +0,14 [0/5] | +0,16 [0/5] |

![Sous-espaces tirés du gradient contre sous-espaces aléatoires](../results/figures/sous_espace_appris.png)

- **Les directions tirées du gradient valent des directions aléatoires** : les écarts (−0,08 à +0,02) sont du même ordre que
  l'écart-type entre graines (0,02 à 0,09 sur Mackey-Glass, NARMA10 et la mémoire, 0,17 sur Hénon, 0,27 sur Lorenz).
- **Les positions tirées du gradient sont pires que des positions au hasard** (3 victoires sur 15 couples tâche × dimension
  pour le plus grand gradient, 0 sur 15 avec l'accord). Elles se concentrent sur quelques neurones :

  | Positions | r | Lignes de W touchées (neurones qui reçoivent) | Part des positions dans les 10 lignes les plus touchées | Colonnes touchées |
  |---|---|---|---|---|
  | Plus grand gradient | 100 | 17 | 90 % | 36 |
  | Accord | 100 | 22 | 80 % | 32 |
  | Au hasard | 100 | 64 | 29 % | 64 |
  | Plus grand gradient | 1 000 | 59 | 45 % | 79 |
  | Accord | 1 000 | 72 | 37 % | 83 |
  | Au hasard | 1 000 | 100 | 16 % | 100 |

  Le gradient de départ se concentre sur les lignes de quelques neurones, et les positions qu'il choisit avec ; des positions
  au hasard modifient presque tous les neurones. C'est une corrélation, compatible avec « mieux vaut toucher un peu tous les
  neurones que beaucoup quelques-uns », sans la prouver. Les positions choisies ne privilégient pas les coefficients non nuls
  de `W₀` (10 % d'entre elles, comme au hasard).
- **La dimension compte, la forme non.** De r = 100 à 1 000, la plupart des sous-espaces gagnent 0,2 à 0,5 en log10,
  contrôles compris. À r = 1 000, des positions au hasard font aussi bien que la bande k = 10 (−0,09 en moyenne, 4 tâches sur
  5), ce qui prolonge le contrôle de la partie 5.5.

**Une décomposition de Lie apprise sur la tâche.** Toutes les décompositions des parties précédentes (SVD, K·A·N, Schur) sont
des découpages de `gl(n)` choisis pour leur élégance mathématique, pas pour leur alignement avec la perte. Le « bon »
découpage serait déterminé par la géométrie de la perte vue depuis la matrice de départ, et cette géométrie se sonde :

1. depuis le réservoir de départ, on calcule pour chaque lot le vecteur tangent `X_k = M⁻¹G_k` (le flot de gradient
   multiplicatif, `W = W₀·exp(X)`, donc `ΔW = W₀·X` au premier ordre), avec `M = W₀` ;
2. on forme la covariance des tangents `C = E[X_k ⊗ X_k]` ;
3. on la diagonalise : ses directions propres principales sont les directions que la perte veut explorer, et la hiérarchie
   des valeurs propres (les raideurs) dit dans quel ordre dégeler les blocs.

`M = W₀` est presque singulière (plus petite valeur singulière de l'ordre de 2·10⁻⁴ à 100 neurones) : `M⁻¹` est régularisée,
`X = (MᵀM + λI)⁻¹MᵀG` avec `λ = (ε·σ_max)²`, pour ε = 0,01 et 0,1 ; le gradient brut (`X = G`) sert de troisième variante. La
covariance (non centrée) est estimée sur 2 560 fenêtres d'entraînement. On compare, à d = 100, 300 et 1 000 coefficients, les
d premières directions propres, les d suivantes (rangs 1 000 et au-delà) et d directions aléatoires ; puis on dégèle les
1 000 premières directions en dix blocs de 100, un bloc tous les ~136 pas, dans l'ordre des valeurs propres décroissantes,
dans un ordre aléatoire, ou dans l'ordre croissant. Cinq tâches, 3 graines.

**La covariance des tangents est presque de rang un :**

| Tâche | Tangent | Part de la trace dans la 1ʳᵉ direction | dans les 5 premières | λ₂ / λ₁ | λ₅ / λ₁ |
|---|---|---|---|---|---|
| Mackey-Glass | `G` brut | 0,80 | 1,00 | 0,23 | 0,003 |
| Mackey-Glass | `M⁻¹G`, ε = 0,01 | 0,85 | 1,00 | 0,16 | 0,002 |
| NARMA10 | `G` brut | 0,81 | 0,97 | 0,10 | 0,015 |
| NARMA10 | `M⁻¹G`, ε = 0,01 | 0,83 | 0,98 | 0,11 | 0,011 |
| Mémoire | `G` brut | 0,35 | 0,91 | 0,80 | 0,13 |
| Mémoire | `M⁻¹G`, ε = 0,01 | 0,43 | 0,95 | 0,67 | 0,07 |
| Lorenz | `G` brut | 0,71 | 0,99 | 0,28 | 0,010 |
| Lorenz | `M⁻¹G`, ε = 0,01 | 0,72 | 0,99 | 0,28 | 0,010 |
| Hénon | `G` brut | 0,44 | 0,93 | 0,50 | 0,15 |
| Hénon | `M⁻¹G`, ε = 0,01 | 0,51 | 0,95 | 0,40 | 0,11 |

Deux à cinq directions portent 90 % de la trace ; la centième valeur propre est plus de 10 000 fois plus petite que la
première. La « hiérarchie de raideurs » se réduit donc à quelques directions : le premier bloc de 100 contient toute la partie
structurée, et les neuf autres ne sont distingués que par le bruit d'estimation. Ce n'est pas seulement le gradient moyen : sur
les tâches plantées de 9.6, la covariance centrée garde la même part de trace dans sa première direction que la non centrée, à
0,01 près.

Écarts moyens (log10 NMSE, négatif = la première ligne est mieux ; moyenne sur les cinq tâches, [victoires]) :

| Tangent | Comparaison | d = 100 | d = 300 | d = 1 000 |
|---|---|---|---|---|
| `G` brut | premières − aléatoires | +0,01 [3/5] | −0,04 [2/5] | −0,01 [2/5] |
| | premières − suivantes | +0,02 [3/5] | −0,03 [3/5] | +0,00 [2/5] |
| `M⁻¹G`, ε = 0,01 | premières − aléatoires | +0,23 [1/5] | +0,16 [1/5] | +0,13 [0/5] |
| | premières − suivantes | +0,20 [1/5] | +0,15 [0/5] | +0,19 [0/5] |
| `M⁻¹G`, ε = 0,1 | premières − aléatoires | +0,06 [1/5] | +0,08 [2/5] | +0,03 [2/5] |
| | premières − suivantes | +0,06 [1/5] | +0,10 [2/5] | +0,05 [2/5] |

| Tangent | Dégel décroissant − ordre aléatoire | Dégel décroissant − ordre croissant | Dégel décroissant − mêmes directions sans dégel |
|---|---|---|---|
| `G` brut | +0,07 [0/5] | −0,03 [1/5] | −0,01 |
| `M⁻¹G`, ε = 0,01 | −0,02 [4/5] | +0,04 [2/5] | −0,02 |
| `M⁻¹G`, ε = 0,1 | −0,07 [4/5] | +0,04 [2/5] | −0,04 |

![Premières directions propres, suivantes et aléatoires, pour chaque tangent](../results/figures/protocole_lie.png)

- **En moyenne, les premières directions propres ne battent ni les suivantes ni des directions aléatoires.** Avec le
  gradient brut, les écarts sont nuls ; avec le tangent multiplicatif, les premières directions sont moins bonnes (+0,03 à
  +0,23), et la régularisation forte (ε = 0,1) ramène presque au gradient brut.
- **L'ordre de dégel ne compte pas** : les écarts (−0,07 à +0,07) sont sous l'écart-type entre graines (0,04 à 0,24 selon la
  tâche), et le dégel progressif ne fait pas mieux que d'apprendre les 1 000 directions d'emblée.
- **Mais la moyenne cachait deux groupes de tâches.** Avec le gradient brut, sur les deux tâches où la bande ne gagne pas, les
  premières directions battaient les directions aléatoires, les suivantes, et même la bande k = 10 à coefficients égaux :

  | Tâche | Meilleure structure fixée (9.2) | Premières − aléatoires (d = 100, 300, 1 000) | Premières − suivantes | Premières (d = 1 000) − bande k = 10 |
  |---|---|---|---|---|
  | Mackey-Glass | bande k = 10 | +0,14 ; +0,06 ; +0,22 | +0,35 ; +0,06 ; +0,16 | +0,19 |
  | NARMA10 | bande k = 10 | −0,02 ; +0,04 ; +0,05 | −0,03 ; −0,04 ; +0,10 | +0,16 |
  | Mémoire | bande k = 10 | +0,12 ; +0,07 ; +0,09 | +0,08 ; +0,08 ; +0,17 | +0,13 |
  | Lorenz | RNN | −0,15 ; −0,29 ; −0,27 | −0,23 ; −0,05 ; −0,18 | −0,19 |
  | Hénon | tourné | −0,01 ; −0,10 ; −0,13 | −0,06 ; −0,19 ; −0,23 | −0,10 |

  Le même motif se retrouve avec les directions principales des sous-espaces libres (−0,20, −0,06, −0,17 sur Lorenz ; +0,01,
  −0,19, −0,13 sur Hénon ; positif sur les trois autres tâches), mais sur les mêmes réservoirs : ce n'est pas une confirmation
  indépendante. C'était le premier résultat de la partie où le gradient semblait dire quelque chose qu'une structure fixée ne
  dit pas. Il a été remarqué après coup ; il est vérifié sur des graines neuves en 9.7, et il n'y tient pas.

## 9.6 Des tâches dont la structure est connue

Les sections précédentes montrent que le gradient ne dit pas quelle structure apprendre sur nos tâches. Mais rien ne dit que
ces tâches aient une structure à trouver. Ici, la structure est plantée dans un enseignant, et atteignable par construction :
si le gradient ne la retrouve pas même là, la conclusion devient générale ; s'il la retrouve, l'échec précédent vient des
tâches.

**Les enseignants.** Un réseau de 100 neurones, avec le même `W_in` que l'élève et des biais nuls, dont la matrice s'écarte du
réservoir de départ `W₀` de l'élève d'une distance fixée d'avance, `‖W_ens − W₀‖_F = a·‖W₀‖_F` :

| Tâche | Matrice de l'enseignant | Structure attendue |
|---|---|---|
| T1 tournée | `e^{cΩ} W₀ e^{−cΩ}`, Ω antisymétrique aléatoire | tourné |
| T2 cisaillée | `Q(R + cS)`, S strictement triangulaire aléatoire | cisaillé plein |
| T3 accordée | `U diag(σ ⊙ e^{cz}) Vᵀ`, z aléatoire | accordé |
| T4 bande | `Q(R + cS)`, S limitée aux 10 premières sur-diagonales | bande k = 10 |
| T5 témoin | `W₀ + cP`, P pleine aléatoire | RNN |

(Q, R : QR de `W₀` ; U, σ, V : sa SVD, avec les conventions des élèves.) L'entrée est i.i.d., `u_t ~ N(0 ; 0,5²)`, sur 4 000
pas ; la sortie est une lecture aléatoire fixée de l'état de l'enseignant, centrée réduite, plus un bruit de mesure à 30 dB
(plancher de NMSE : 10⁻³, soit −3 en log10). L'élève prédit `y_t` à partir de `u_≤t`. Huit structures apprennent ; les
signaux choisissent parmi les six structures récurrentes autres que le RNN. L'oracle (réservoir figé à la matrice de
l'enseignant, lecture seule) atteint le plancher du bruit dans tous les cas (−2,95 à −3,05) : la construction est correcte.

**Deux premières amplitudes, a = 0,1 et a = 0,5** (1 500 pas, sans arrêt anticipé, 3 graines). Le critère, fixé avant de
voir les résultats : un signal est « O » s'il retrouve la structure plantée dans au moins 6 des 8 cas (T1 à T4 × deux
amplitudes), « partiel » de 3 à 5, « N » sinon ; le hasard en trouve 1,3.

| Cas | Oracle | Réservoir classique | Meilleure structure | Structure attendue | L'attendue gagne-t-elle ? |
|---|---|---|---|---|---|
| T1, a = 0,1 | −2,99 | −2,93 | tourné (−2,66) | tourné | oui |
| T2, a = 0,1 | −3,01 | −2,80 | tourné (−2,69) | cisaillé plein (−2,61) | non |
| T3, a = 0,1 | −2,95 | −2,76 | tourné (−2,70) | accordé (−2,66) | non |
| T4, a = 0,1 | −2,98 | −2,86 | bande k = 10 (−2,74) | bande k = 10 | oui |
| T5, a = 0,1 | −3,00 | −2,81 | tourné (−2,72) | RNN (−2,61) | non |
| T1, a = 0,5 | −2,99 | −2,44 | tourné (−2,62) | tourné | oui |
| T2, a = 0,5 | −2,99 | −1,29 | bande k = 10 (−1,87) | cisaillé plein (−1,80) | non |
| T3, a = 0,5 | −2,99 | −0,45 | accordé (−0,93) | accordé | oui |
| T4, a = 0,5 | −2,97 | −1,73 | bande k = 10 (−2,18) | bande k = 10 | oui |
| T5, a = 0,5 | −3,00 | −1,26 | accordé (−1,63) | RNN (−1,31) | non |

![Vérité terrain des tâches plantées, et structure plantée retrouvée par chaque signal](../results/figures/enseignants.png)

- **À a = 0,1, l'entraînement dégrade.** Le réservoir classique, figé en `W₀`, est déjà à 0,06 à 0,2 du plancher, et toutes
  les structures entraînées font *moins bien* que lui (de 0,06 à 0,5). Or le réservoir figé est dans l'espace de chaque
  structure, à paramètres nuls : ce n'est pas la géométrie qui manque, c'est l'entraînement (1 500 pas d'Adam sans arrêt
  anticipé) qui ajoute son propre bruit.
- **À a = 0,5, la structure plantée gagne trois tâches sur quatre** (T1, T3, T4), mais aucune structure n'approche l'oracle :
  la meilleure reste de 0,4 à 2,1 au-dessus du plancher. Sur T2, la bande k = 10, un sous-espace du cisaillé plein, bat le
  cisaillé plein ; sur T5, sans structure, l'accordé gagne et le RNN est dernier.

| Signal (structure plantée retrouvée sur 8 cas) | Complet (W, `W_in`, b) | Récurrent seul (W) |
|---|---|---|
| Part captée | 0 / 8 | 0 / 8 |
| Part par paramètre | 2 / 8 | 4 / 8 |
| Signal sur bruit | 3 / 8 | 3 / 8 |
| Accord entraînement-validation | 2 / 8 | 3 / 8 |
| Pas de Newton | 0 / 8 | 0 / 8 |
| Directions principales | 0 / 8 | 0 / 8 |
| PCA de la covariance des tangents (alignement de la 1ʳᵉ direction / hasard), `G` brut | 1 / 8 | |
| PCA, `M⁻¹G` (ε = 0,1) | 3 / 8 | |
| Taille | 0 / 8 | |

- **Aucun signal n'obtient « O ».** Les signaux qui suivent la taille choisissent le nilpotent, le plus grand des six, dans
  presque tous les cas. La première direction propre de la PCA n'est pas plus dans l'espace tangent de la structure attendue
  que ne le voudrait le hasard (rapport de 0,6 à 1,1).
- **Les succès partiels ne tiennent pas d'une graine à l'autre.** Sur la première graine complète, le signal sur bruit
  récurrent retrouvait 4 cas, dont le tourné, le cisaillé et l'accordé à a = 0,1 ; avec trois graines, il en retrouve 3, et ce
  ne sont plus les mêmes. L'information propre à la tâche n'est nulle part significative après correction pour seize
  signaux.
- **Mais aucun des deux régimes n'est celui où la question a une réponse** : à a = 0,1 la tâche est déjà résolue par le
  réservoir figé, à a = 0,5 la structure plantée n'est pas atteinte. C'est l'objet de la section suivante.

## 9.7 Les deux indices, vérifiés sur des graines neuves

Deux questions restaient ouvertes : le régime intermédiaire des tâches plantées, où la structure compte et peut être apprise,
et les premières directions du gradient sur Lorenz et Hénon. Les deux sont reprises sur des graines qui n'avaient servi à
rien, avec un critère fixé avant de lancer le calcul.

**L'entraînement.** Pour toutes les tâches plantées, le même budget pour toutes les structures : 3 000 pas, erreur de
validation tous les 100 pas et retour au meilleur point (arrêt anticipé).

**Le régime intermédiaire.** Sur une graine à part (graine 3), on cherche le plus petit a de {0,2 ; 0,3 ; 0,4} tel que, sur
chacune des tâches T1 à T4, (i) le réservoir figé est à plus de 0,5 du plancher, et (ii) la structure plantée bat chacune des
autres d'au moins 0,1. **Aucune amplitude ne satisfait les deux conditions** :

| Condition qui rate (nombre de tâches sur 4) | a = 0,2 | a = 0,3 | a = 0,4 |
|---|---|---|---|
| (i) réservoir figé trop près du plancher | 3 | 2 | 1 |
| (ii) structure plantée qui ne gagne pas de 0,1 | 4 | 4 | 3 |
| Avance moyenne de la structure plantée | −0,02 | +0,01 | +0,04 |

C'est presque toujours (ii) qui rate : même quand il y a de quoi apprendre, la structure plantée ne bat pas les autres. Sur
T1, à a = 0,2 et 0,3, l'arrêt anticipé revient au pas 0 pour toutes les structures : aucun entraînement n'améliore la
validation par rapport au réservoir figé. On garde l'amplitude où la structure plantée a la plus grande avance, a = 0,4, et
la suivante, a = 0,5.

**Sur les graines de test, la structure plantée ne gagne toujours pas** (graines 4, 5, 6 ; avance de la structure plantée
sur la meilleure des autres, en log10 ; positif = elle gagne) :

| Cas | Réservoir classique − oracle | Meilleure structure | Avance de la plantée |
|---|---|---|---|
| T1 tournée, a = 0,4 | +0,22 | bande k = 3 | −0,06 |
| T1 tournée, a = 0,5 | +0,56 | tourné | +0,03 |
| T2 cisaillée, a = 0,4 | +0,82 | bande k = 10 | −0,07 |
| T2 cisaillée, a = 0,5 | +0,99 | bande k = 10 | −0,06 |
| T3 accordée, a = 0,4 | +1,03 | bande k = 10 | −0,01 |
| T3 accordée, a = 0,5 | +1,48 | bande k = 10 | −0,10 |
| T4 bande, a = 0,4 | +1,26 | bande k = 10 | +0,02 |
| T4 bande, a = 0,5 | +1,44 | bande k = 3 | −0,00 |
| T5 témoin, a = 0,4 | +0,71 | bande k = 10 | −0,08 |
| T5 témoin, a = 0,5 | +1,19 | bande k = 3 | −0,04 |

Sur sept des huit cas plantés, il y a de quoi apprendre (le réservoir figé est à plus de 0,5 du plancher), et pourtant la
structure plantée ne gagne nettement dans aucun. Ce sont les bandes, de 294 à 945 coefficients, qui gagnent neuf cas sur dix,
y compris contre la structure exacte : sur T3 à a = 0,5, l'accordé, qui contient exactement l'enseignant avec 100 paramètres,
finit 0,10 derrière la bande k = 10.

![Les deux indices sur des graines neuves](../results/figures/preenregistre.png)

**Le signal sur bruit.** On s'attendait à ce que le signal sur bruit du gradient, restreint à la récurrence, retrouve la
structure plantée dans au moins 5 des 8 cas (T1 à T4, a = 0,4 et 0,5 ; le hasard en trouve 1,3, et 5 ou plus par hasard a
une probabilité d'environ 0,5 %). **On ne l'a pas : 2 cas sur 8.** Il retrouve le tourné et le cisaillé plein à a = 0,4,
rien à a = 0,5 ; ailleurs il choisit surtout le cisaillé plein ou la bande k = 3, quelle que soit la structure plantée. Aucun
autre signal ne dépasse 2 cas sur 8. De toute façon, la structure plantée ne gagnant nettement aucun cas, un signal qui la
retrouverait n'aurait rien fait gagner.

**Les premières directions du gradient.** On s'attendait à ce que, avec d = 1 000 et le protocole de 9.5, les premières
directions propres du gradient battent la bande k = 10 sur Lorenz et Hénon, et fassent moins bien que des directions
aléatoires sur les trois autres tâches. **On ne l'a pas sur Lorenz et Hénon** (graines 3, 4, 5) :

| Tâche | Premières − bande k = 10 | Premières − aléatoires | Attendu | Obtenu ? |
|---|---|---|---|---|
| Mackey-Glass | +0,13 | +0,16 | moins bonnes que l'aléatoire | oui |
| NARMA10 | +0,11 | +0,03 | moins bonnes que l'aléatoire | oui, de peu |
| Mémoire | +0,13 | +0,09 | moins bonnes que l'aléatoire | oui |
| Lorenz | +0,22 | −0,16 | meilleures que la bande | non |
| Hénon | +0,01 | +0,05 | meilleures que la bande | non |

Sur des graines neuves, les premières directions du gradient ne battent la bande ni sur Lorenz ni sur Hénon : l'indice de 9.5
venait des graines 0 à 2. (La graine 3 sert à la fois au réglage de l'amplitude et à cette vérification, mais sur des tâches
différentes, plantées d'un côté et réelles de l'autre.)

**Description, sans test** (graines 3 à 5) :

| Tâche | Part du déplacement d'un RNN entraîné dans les 1 000 premières directions (hasard : 0,10) | Rang stable de ΔW | Écart à la normalité de `W₀ + ΔW` (Henrici ; `W₀` : 0,73) | Ratio de participation de la covariance |
|---|---|---|---|---|
| Mackey-Glass | 0,40 | 4,0 | 0,72 | 1,3 |
| NARMA10 | 0,25 | 6,0 | 0,70 | 1,7 |
| Mémoire | 0,26 | 3,8 | 0,73 | 4,8 |
| Lorenz | 0,45 | 2,2 | 0,74 | 1,8 |
| Hénon | 0,35 | 2,7 | 0,77 | 3,5 |

Ce qu'apprend le RNN complet est de plus faible rang sur Lorenz et Hénon (rang stable de 2,2 et 2,7, contre 3,8 à 6,0), et
les premières directions du gradient captent 25 à 45 % de son déplacement, quatre fois plus que le hasard. L'écart à la
normalité ne distingue pas les tâches. Comme l'effet attendu n'est pas là, il n'y a rien à expliquer.

## 9.8 Bilan de la partie 9

**Ce que la partie établit.**

| Question | Réponse, à 100 neurones |
|---|---|
| Le gradient au départ classe-t-il les structures ? | non, pas mieux que leur taille ; le choix naïf gagne |
| Et plus tard dans l'entraînement ? | non pour le choix ; l'information propre à la tâche qui apparaît à t = 100 porte sur le réservoir à biais et entrées et le RNN, pas sur les familles ; l'enjeu du choix fond avec t |
| Le gradient désigne-t-il un sous-espace ? | non : des directions qu'il désigne valent des directions aléatoires, des positions qu'il désigne sont pires que des positions au hasard |
| Une décomposition de Lie apprise (PCA du tangent `M⁻¹G`) ? | non : la covariance est presque de rang un, l'ordre de dégel ne compte pas, et l'indice sur Lorenz et Hénon ne tient pas sur des graines neuves |
| Le gradient retrouve-t-il une structure plantée dans la tâche ? | non, dans aucun des quatre régimes testés |
| L'entraînement la retrouve-t-il ? | rarement : à budget égal et avec arrêt anticipé, la structure plantée ne gagne nettement aucun cas des graines de test ; les bandes gagnent |

**Le résultat principal n'est pas celui qui était cherché : la facilité d'apprentissage bat la géométrie de la cible, même
quand la solution exacte est dans la famille.** La question était de savoir si le gradient voit la structure ;
la réponse la plus nette porte sur la vérité terrain elle-même. Quand une structure est plantée dans la tâche et que l'élève
qui la porte contient exactement l'enseignant, ce n'est pas lui qui apprend le mieux en un temps fini : ce sont les bandes,
quelques centaines de coefficients près de la diagonale de la base Q. Ce qui décide de la structure gagnante n'est donc pas la
géométrie de la cible, mais la facilité d'apprentissage : peu de coefficients, chacun avec un effet local et bien
conditionné. Un signal qui chercherait la géométrie de la cible poserait, dans ces conditions, une question à laquelle
l'entraînement ne répond pas. C'est cohérent avec tout le reste du rapport : la partie 5.5 montrait déjà que des coefficients
au hasard font aussi bien que la bande, et la section 9.5 qu'aucun choix de directions ne bat le hasard à dimension égale.

**Ce qui reste solide d'un bout à l'autre** : sur les tâches où la bande gagne, la performance dépend du nombre de
coefficients appris dans la récurrence (de 100 à 1 000, 0,2 à 0,5 en log10), pas de leur emplacement.

**Une méthode, autant qu'un résultat.** Deux indices ont émergé en chemin, et les deux ont été suivis jusqu'à un test qui
pouvait les faire tomber :
- le **signal sur bruit** du gradient : indice au départ (r = 0,34), information propre à la tâche à t = 100 (mais pas sur les
  familles), 4 cas sur 8 sur la première graine des tâches plantées, 3 sur trois graines, puis 2 sur 8 sur des graines
  neuves ;
- les **premières directions du gradient sur Lorenz et Hénon** : repérées après coup, retrouvées sur les mêmes réservoirs
  avec une autre estimation, puis absentes sur des graines neuves.

Chaque fois, l'indice s'est affaibli à chaque nouvelle mesure indépendante, jusqu'à disparaître. Sans les graines neuves,
la partie aurait pu conclure à deux effets qui n'existent pas.

**Les limites.**
- 100 neurones, 3 graines par expérience : les écarts entre méthodes (moins de 0,1 en moyenne) sont du même ordre que
  l'écart-type entre graines. Les conclusions négatives portent sur des effets de cette taille ; un effet bien plus petit
  passerait inaperçu.
- Les signaux sont du premier ordre, ou du second pour le pas de Newton ; un signal qui intègre l'entraînement (l'entraînement
  pilote) fait mieux au tout début de l'entraînement, mais coûte un entraînement.
- Le tangent multiplicatif est régularisé ; la régularisation choisie (ε = 0,01 et 0,1) n'est qu'un choix parmi d'autres.

**Le lien avec l'existant.** Que la descente de gradient se fasse dans un petit sous-espace engendré par les premières
directions de la hessienne est observé sur des réseaux de classification ([Gur-Ari et al., 2018](https://alphaxiv.org/abs/1812.04754)),
avec un sous-espace qui se stabilise après quelques pas ; ici, au départ, ce sous-espace ne compte que quelques directions et
ne suffit pas à guider l'apprentissage. À l'inverse, qu'un sous-espace aléatoire de dimension suffisante suffise à apprendre
une tâche est le résultat de la dimension intrinsèque ([Li et al., 2018](https://alphaxiv.org/abs/1804.08838)), que la
partie 9 retrouve. Ces deux travaux n'ont été lus qu'à travers leur résumé.

---

[Sommaire](../README.md#le-rapport) · [← Partie 8](8_synthese.md) · [Annexe →](annexe_pistes.md)
