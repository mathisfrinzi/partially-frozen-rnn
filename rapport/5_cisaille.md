[Sommaire](../README.md#le-rapport) · [← Partie 4](4_accorde.md) · [Partie 6 →](6_nilpotent.md)

# Partie 5 — Le réservoir cisaillé : apprendre le facteur N de la décomposition d'Iwasawa

![Réservoir cisaillé](../results/figures/schema_cisaille.png)

> **Ce qu'il faut retenir**
>
> - On écrit le réservoir `W₀ = K·A·N₀` (décomposition d'Iwasawa) et on n'apprend que le facteur N, le cisaillement : la rotation K et les amplitudes A restent celles du réservoir. C'est une pièce que l'accordé ne peut pas apprendre.
> - Plein (N(N−1)/2 paramètres), le cisaillement aide à 100 neurones mais s'effondre à 500. **En bande** (k sur-diagonales, k·N paramètres), il bat l'accordé sur les cinq tâches à 500 neurones. Le gain ne tient pas à un k réglé tâche par tâche : un seul k = 10 pour toutes les tâches fait presque aussi bien que le meilleur k de chaque tâche, et le classement tient sur des séries régénérées.
> - **Mais ce n'est pas le cisaillement qui compte.** Le même nombre de coefficients, placés au hasard dans le triangle, ou appris directement dans `W₀` sans passer par la décomposition, fait aussi bien. Le gain vient d'une petite correction creuse de la récurrence, pas de la structure K·A·N.
> - Face aux modèles existants, avec le même budget de recherche : 3,3 fois mieux que le GRU sur Mackey-Glass, 2 fois mieux que le LRU sur NARMA10 ; nette défaite sur Lorenz. Une variante linéaire à porte d'entrée fait 8 fois mieux que le LRU sur NARMA10 et 23 fois mieux sur la mémoire.

L'accordé apprend les amplitudes de la décomposition en valeurs singulières, le tourné ses orientations. Une matrice peut
s'écrire autrement, et chaque écriture sépare d'autres degrés de liberté. Trois décompositions classiques des groupes de
matrices organisent les parties 4 à 6 :

| Partie | Décomposition | Facteurs | Ce qui est appris |
|---|---|---|---|
| 4 · Accordé | Cartan, `K·A·K` (valeurs singulières) | `U`, `diag(σ)`, `Vᵀ` | les amplitudes A |
| **5 · Cisaillé** | **Iwasawa, `K·A·N`** | `Q` orthogonale, `diag(a)` positive, `N₀` unipotente | **le facteur N** (et A dans la variante « avec amplitudes ») |
| 6 · Nilpotent | Jordan-Chevalley | partie semi-simple, partie nilpotente | la partie nilpotente |

Pour une matrice inversible, la décomposition d'Iwasawa est unique : `W₀ = K·A·N₀`, où K est orthogonale, A diagonale à
coefficients strictement positifs, et `N₀` triangulaire supérieure à diagonale unité. N est un produit de **cisaillements**
(transvections : ajouter à une coordonnée un multiple d'une autre). On la calcule par Gram-Schmidt, c'est-à-dire par la
décomposition QR : `K = Q`, `A = diag(a)` avec `a` la diagonale de `R` (rendue positive), et `N₀ = diag(a)⁻¹·R`.

| Modèle | Écriture | Ce qu'on apprend | Paramètres à 100 neurones |
|---|---|---|---|
| Iwasawa, amplitudes (A) | `W = Q·diag(g)·R` | les amplitudes `g` | 300 |
| **Réservoir cisaillé** (N) | `W = Q·(R + S)` | `S`, strictement triangulaire | 5 150 |
| Cisaillé + amplitudes (A et N) | `W = Q·diag(g)·(R + S)` | les deux | 5 250 |

Comme `S` est strictement triangulaire, elle ne touche pas la diagonale de `R`, et `W = Q·(R + S) = K·A·N'` avec
`N' = N₀ + diag(a)⁻¹·S` : seul le facteur N change. L'ensemble des matrices atteignables est exactement l'orbite de `W₀` par le
groupe N. Deux remarques sur ce choix :

- **La paramétrisation est additive sur `R`**, pas sur N. Un même pas sur `S_ij` déplace `N'_ij` de `S_ij / a_i` : les lignes
  où `a_i` est petit sont celles que `S` modifie le plus, relativement à leur contenu de départ. Sur nos réservoirs (500
  neurones, 10 % de connexions), les `a_i` décroissent avec l'indice de ligne, de 0,85 en moyenne sur les 50 premières lignes à
  0,18 sur les 50 dernières (minimum 0,04).
- **Le cas singulier.** Si `W₀` n'est pas inversible, certains `a_i` sont nuls et la décomposition d'Iwasawa ne s'applique
  plus au sens strict. `W = Q·(R + S)` reste défini (le code ne divise jamais par `a`) : c'est alors une extension par QR. Nos
  réservoirs sont inversibles (plus petit `a_i` : 0,004 à 0,009 sur cinq graines, à 100 et 500 neurones) ; un réservoir très
  creux (3 % de connexions à 100 neurones) a 10 à 14 diagonales nulles sur 100.

Au départ, tous ces modèles reproduisent exactement `W₀` (erreur de l'ordre de 10⁻⁷, vérifiée par un test). Comme partout,
`W_in` et les biais apprennent aussi.

## 5.1 Résultats à 100 neurones

Cinq tâches, 3 graines. Facteur d'erreur par rapport au réservoir accordé (×2 = deux fois moins d'erreur, ÷2 = deux fois
plus) ; en gras, au-delà du seuil de bruit.

| Modèle | Paramètres | Mackey-Glass | NARMA10 | Mémoire | Lorenz | Hénon |
|---|---|---|---|---|---|---|
| Réservoir accordé, NMSE | 300 | 9.0e-5 | 0.023 | 0.0029 | 6.3e-4 | 0.042 |
| Iwasawa : amplitudes (A) | 300 | ÷1,2 | ÷1,1 | ÷1,1 | ÷1,3 | **×1,2** |
| **Réservoir cisaillé (N)** | 5 150 | ×1,2 | **×1,2** | **×1,2** | **×4,2** | **×2,0** |
| Cisaillé + amplitudes (A et N) | 5 250 | **×1,4** | ×1,1 | **×1,2** | **×4,4** | ÷1,4 |
| *Partie 3 : réservoir tourné* | 10 200 | **×1,3** | ×1,0 | **÷9,3** | ×1,7 | **×2,8** |
| *RNN* | 10 200 | ×1,1 | **÷1,4** | ×1,0 | **×4,7** | ÷1,6 |

![Décompositions : erreur par rapport à l'accordé](../results/figures/decompositions_n100.png)

- **Le cisaillement aide, et surtout sur Lorenz** : 4,2 fois mieux que l'accordé, presque autant que le RNN (4,7 fois) avec
  deux fois moins de paramètres. Ce que le RNN gagne sur l'accordé à Lorenz semble donc être, pour l'essentiel, de la
  non-normalité : on le retrouve en gardant la rotation et les amplitudes figées et en n'apprenant que la partie triangulaire.
- **C'est le modèle le plus régulier de l'étude** : égal ou meilleur sur les cinq tâches, jamais moins bon, avec un gain net
  sur quatre (NARMA10, mémoire, Lorenz, Hénon).
- **À budget égal (300 paramètres), rien ne bat l'accordé** : les amplitudes d'Iwasawa reviennent au même. Les valeurs
  singulières restent le meilleur choix quand on n'apprend qu'un nombre par direction.
- Ajouter les amplitudes au cisaillement n'aide pas, et fait même perdre le gain sur Hénon.
- Sur la mémoire, le cisaillement aide un peu par rapport aux amplitudes seules (×1,2 contre ÷1,1), ce qui va dans le sens
  d'une amplification transitoire créée par la non-normalité.

## 5.2 Le prix, et la version en bande

Ces gains coûtent N(N−1)/2 paramètres : 5 000 à 100 neurones, 17 fois l'accordé et la moitié du RNN ; à 500 neurones, environ
125 000. Ce n'est plus « apprendre peu ». D'où le **cisaillement en bande** : seules les k premières sur-diagonales
apprennent (k·N paramètres environ ; 30 au lieu de 66 sur l'icône ci-dessus, pour 12 neurones). À 500 neurones, avec `W_in`
et les biais, cela fait 1 499 paramètres pour k = 1 (autant que l'accordé), 2 494 pour k = 3 et 5 945 pour k = 10. Comme le
cisaillé plein, la version en bande n'apprend pas d'amplitudes g.

## 5.3 À 500 neurones : le cisaillement plein s'effondre, la bande gagne

Cinq tâches, 3 graines, 1 500 pas. Facteur d'erreur par rapport au réservoir accordé (×2 = deux fois moins d'erreur ; en gras,
au-delà du seuil de bruit) :

| Modèle (paramètres) | Mackey-Glass | NARMA10 | Mémoire | Lorenz | Hénon |
|---|---|---|---|---|---|
| Réservoir accordé (1 500), NMSE | 7.5e-8 | 0.0102 | 0.00186 | 4.3e-5 | 0.065 |
| Cisaillé plein (125 750) | **÷146** | **÷10** | **÷48** | **×4,6** | **÷3,4** |
| Cisaillé en bande, k = 1 (1 499) | **÷1,8** | ×1,1 | **×1,7** | **×6,1** | **×1,8** |
| Cisaillé en bande, k = 3 (2 494) | ÷1,2 | **×1,6** | **×1,5** | **×6,2** | **×1,5** |
| **Cisaillé en bande, k = 10 (5 945)** | **×1,8** | **×1,3** | **×1,6** | **×15** | **×1,3** |

![Cisaillé et nilpotent à 500 neurones](../results/figures/cisaille_nilpotent_500.png)

- **Le cisaillement plein ne passe pas à l'échelle.** À 500 neurones, ses 125 000 paramètres font 10 à 146 fois plus d'erreur
  que l'accordé, sauf sur Lorenz, où il garde son gain (4,6 fois). La courbe en k (section 5.4) montre l'écart entre erreur
  d'entraînement et erreur de test qui accompagne cet effondrement.
- **La bande garde le gain et supprime l'effondrement.** Avec k = 10, le cisaillé bat l'accordé sur les cinq tâches, avec un
  gain très net sur Lorenz (15 fois : 3,3e-6, autant que l'accordé de rang 8 et ses 9 500 paramètres, et 13 fois mieux que le
  GRU de 60 unités, 11 340 paramètres). Avec k = 1 ou 3, il bat l'accordé sur quatre tâches et perd un peu sur Mackey-Glass.
- **Il gagne sans apprendre d'amplitudes** : le seul changement apporté à la récurrence est un couplage triangulaire entre
  directions voisines de la base `Q`. La section 5.5 montre que la forme de ce couplage ne compte pas.


## 5.4 La courbe en k : un seul k pour toutes les tâches

Le gain de la bande pourrait ne tenir qu'à un k bien choisi pour chaque tâche. Pour le vérifier, k va de 0 (seuls les biais
et `W_in` apprennent) au cisaillement plein, sur les cinq tâches, à 500 neurones, 3 graines, avec l'erreur d'entraînement et
celle de test. Facteur d'erreur de test par rapport à l'accordé (×2 = deux fois moins d'erreur ; en gras, au-delà du seuil de
bruit) :

| k (paramètres) | Mackey-Glass | NARMA10 | Mémoire | Lorenz | Hénon |
|---|---|---|---|---|---|
| Réservoir accordé (1 500), NMSE | 7.4e-8 | 0.0102 | 0.00186 | 4.2e-5 | 0.064 |
| k = 0, biais et entrées seuls (1 000) | **÷3,0** | **÷2,6** | ×1,0 | **×2,8** | **×2,5** |
| k = 1 (1 499) | **÷1,8** | ×1,1 | **×1,7** | **×6,0** | **×1,8** |
| k = 3 (2 494) | ÷1,1 | **×1,6** | **×1,4** | **×6,2** | **×1,5** |
| k = 10 (5 945) | **×1,8** | **×1,3** | **×1,6** | **×15** | **×1,3** |
| k = 20 (10 790) | **×1,9** | **÷1,3** | **×1,3** | **×13** | **÷1,4** |
| k = 50 (24 725) | **÷2,4** | **÷2,9** | ÷1,1 | **×15** | **÷2,7** |
| k = 100 (45 950) | **÷16** | **÷5,2** | **÷2,4** | **×8,5** | **÷2,6** |
| plein (125 750) | **÷148** | **÷10** | **÷48** | **×4,6** | **÷3,4** |
| k = 10 avec amplitudes (6 445) | **×2,2** | **÷1,1** | **×1,3** | **×13** | **÷3,5** |

![Courbe en k](../results/figures/shear_curve.png)

- **La courbe a un plateau, de k = 2 à k = 10**, sur toutes les tâches. Un seul k = 10 pour les cinq tâches bat l'accordé
  partout et fait en moyenne 2,4 fois moins d'erreur que lui (−0,38 en log10). Choisir le meilleur k de chaque tâche, sur le
  test, ne gagne que 0,05 de plus : **le gain ne vient pas d'un k réglé tâche par tâche.**
- **Au-delà de k = 20, le modèle sur-apprend.** L'erreur d'entraînement continue de baisser quand k grandit, celle de test
  remonte : sur NARMA10, le rapport test/entraînement passe de 2,4 (k = 1) à 5,8 (k = 10), 46 (k = 50) et 618 (plein) ; même
  allure sur la mémoire et Hénon : l'effondrement du cisaillé plein (section 5.3) s'accompagne d'un sur-apprentissage.
  Mackey-Glass fait exception : l'erreur d'entraînement y remonte aussi au-delà de k = 20.
- **Lorenz** est la seule tâche où presque tout k aide beaucoup (6 à 15 fois), plein compris.
- **Hénon** est la seule où la bande ne bat pas le réservoir à biais et entrées appris (k = 0), meilleur que tous les k.
- **Les amplitudes** n'ajoutent rien de régulier : elles aident sur Mackey-Glass, nuisent fortement sur Hénon.

## 5.5 Contrôles : ce n'est pas le cisaillement qui compte

Deux questions restent : la **bande** compte-t-elle, ou seulement le nombre de coefficients ? Et la **décomposition**
compte-t-elle, ou une correction quelconque de même taille ferait-elle aussi bien ? Trois contrôles, avec exactement le même
nombre de coefficients appris que la bande k (mêmes réservoirs, mêmes graines) :

- **cisaillé creux** : les coefficients de `S` sont placés au hasard dans tout le triangle de `R`, au lieu des k premières
  sur-diagonales ;
- **dans `W₀`, au hasard** : `W = W₀ + M`, avec `M` creuse, à des positions tirées au hasard dans toute la matrice, dans la
  base des neurones, sans aucune décomposition ;
- **dans le triangle de `W₀`** : idem, positions tirées dans le triangle supérieur de `W₀` lui-même (pas dans celui de `R`).

Facteur d'erreur par rapport à l'accordé, k = 10 (5 945 paramètres pour tous) :

| Contrôle (k = 10) | Mackey-Glass | NARMA10 | Mémoire | Lorenz | Hénon |
|---|---|---|---|---|---|
| Cisaillé en bande | **×1,8** | **×1,3** | **×1,6** | **×15** | **×1,3** |
| Cisaillé creux (positions au hasard dans le triangle de `R`) | **×1,8** | **×1,5** | **×1,6** | **×21** | **×1,6** |
| Coefficients appris dans `W₀`, au hasard | **×1,5** | **×1,6** | **×1,7** | **×5,2** | **×1,8** |
| Coefficients appris dans le triangle de `W₀` | **×1,4** | **×1,5** | **×1,7** | **×13** | **×2,9** |

Avec k = 3 (2 494 paramètres), même tableau : les quatre versions sont à moins de 0,15 en log10 les unes des autres sur
Mackey-Glass, NARMA10 et la mémoire, et les contrôles dans `W₀` font mieux que la bande sur Hénon.

- **La bande ne compte pas** : des positions au hasard dans le triangle font aussi bien, et un peu mieux sur NARMA10, Lorenz
  et Hénon.
- **La décomposition ne compte pas non plus.** Apprendre le même nombre de coefficients directement dans `W₀` fait aussi bien
  que le cisaillé sur Mackey-Glass, NARMA10 et la mémoire, et mieux sur Hénon (2,9 fois l'accordé pour le triangle de `W₀`,
  contre 1,3 pour la bande). Seul Lorenz garde un avantage à la base `Q` sur les positions au hasard dans `W₀` (15 contre 5,2
  fois), mais le triangle de `W₀` l'égale (13 fois).
- **Ce qui compte, c'est d'apprendre quelques milliers de coefficients récurrents, épars**, en plus des biais et des entrées.
  C'est une **correction creuse** de la récurrence, proche du réservoir retouché de la partie 1 (qui apprend les connexions
  existantes), avec moins de coefficients et à des positions quelconques.

## 5.6 Sur des données fraîches

Les modèles et leurs réglages ont été fixés avant la mesure ; les six tâches sont régénérées avec d'autres graines de
génération (pour Mackey-Glass, un segment de la trajectoire jamais utilisé). 500 neurones, 3 graines. Facteur d'erreur par
rapport à l'accordé :

| Modèle (paramètres) | Mackey-Glass | NARMA10 | Mémoire | Lorenz | Hénon | Wiener-Hammerstein |
|---|---|---|---|---|---|---|
| Réservoir accordé (1 500), NMSE | 4.2e-8 | 0.0088 | 0.0021 | 4.4e-5 | 0.047 | 0.0015 |
| Biais et entrées seuls, k = 0 (1 000) | **÷3,0** | **÷1,9** | ×1,1 | ×3,2 | **×2,5** | **÷1,7** |
| Cisaillé, k = 3 (2 494) | ×1,0 | **×1,8** | **×1,5** | ×2,1 | **×1,5** | **×1,5** |
| Cisaillé, k = 10 (5 945) | **×1,9** | **×1,9** | **×1,7** | ×1,6 | **×1,5** | ×1,0 |
| Cisaillé à porte d'entrée, k = 10 (6 945) | **×1,9** | **×2,1** | **×1,9** | ×1,9 | **×1,9** | **×1,2** |
| GRU, 100 unités (30 900) | **÷300** | **×1,3** | **÷220** | ×7,9 | **×26** | **×1,6** |
| GRU, 500 unités (754 500) | ×1,0 | **÷16** | **÷300** | **×420** | **×160** | **×2,1** |
| LRU, 200 unités (40 700) | **÷6 900** | ×1,0 | ÷2,8 | **÷2 000** | **×2,5** | **÷9,8** |

![Données fraîches](../results/figures/shear_fresh.png)

- **Contre l'accordé, le classement tient** : k = 10 gagne sur quatre tâches sur six et égale les deux autres (Lorenz, dont
  l'écart est bruité, et Wiener-Hammerstein) ; k = 3 gagne sur quatre et égale deux.
- **Contre le GRU, rien ne change** : le GRU de 500 unités, sans réglage, domine Lorenz et Hénon (420 et 160 fois l'accordé)
  et gagne sur Wiener-Hammerstein ; il est très mauvais sur NARMA10 et la mémoire, comme dans les parties précédentes.
- Sur Hénon, le réservoir à biais et entrées appris reste devant toutes les bandes, comme en 5.4.
- **EMPS**, un système réel jamais utilisé (table de positionnement avec frottement sec et un intégrateur, mesurée en boucle
  fermée) : **tous les modèles échouent** (NMSE supérieure à 1, pire que de prédire la moyenne, pour tous, GRU compris). Le
  protocole de simulation pure ne convient pas à ce système ; aucune conclusion n'en est tirée.

## 5.7 À budget égal contre le GRU

Le cisaillé en bande ajouté à la courbe à budget égal de la partie 4.9 (mêmes graines et mêmes réservoirs ; GRU, LRU et
accordé de rang r repris tels quels). N = 500 avec k = 1 à 100, et N = 100 ou 200 avec k = 10 pour les petits budgets.

![Budget égal](../results/figures/shear_budget.png)

| Tâche | Cisaillé en bande | GRU de taille voisine | Plus grand GRU |
|---|---|---|---|
| Mackey-Glass | 4.0e-8 (k = 10, 5 945 par.) | 6.1e-5 (60 unités, 11 340) | 1.1e-7 (500 unités, 754 500) : **le cisaillé fait 2,8 fois mieux** |
| NARMA10 | 0.0065 (k = 3, 2 494) | 0.0062 (60 unités, 11 340) | 0.154 (500 unités) |
| Mémoire | 0.0011 (k = 1, 1 499) | 0.0034 (60 unités) | 0.55 (500 unités) |
| Lorenz | 2.0e-6 (k = 30, 15 535) | 6.1e-6 (100 unités, 30 900) | **2.2e-7 (500 unités)** |
| Hénon | 0.018 (N = 100, k = 10, 1 145) | 0.018 (100 unités, 30 900) | **8.4e-4 (500 unités)** |
| Wiener-Hammerstein | 8.4e-4 (N = 100, k = 10, 1 145) | 0.0012 (20 unités, 1 380) ; 6.4e-4 (60 unités) | **3.9e-4 (300 unités)** |

- **À budget égal, le cisaillé bat le GRU sur Mackey-Glass, NARMA10 et la mémoire**, et sur Mackey-Glass il bat même le plus
  grand GRU, avec 127 fois moins de paramètres. Sur NARMA10, il égale le GRU de 60 unités (le seul qui réussit cette tâche)
  avec 4,5 fois moins de paramètres.
- **Sur Lorenz**, il bat les GRU jusqu'à 100 unités (3 fois mieux que celui de 30 900 paramètres), mais au-delà de 100 000
  paramètres le GRU reprend l'avantage (9 fois mieux à 500 unités).
- **Sur Hénon**, il égale un GRU 27 fois plus gros ; mais le meilleur de nos modèles y reste la correction de rang r sans g
  (partie 4.9), 12 à 17 fois meilleure que la bande.
- **Sur Wiener-Hammerstein**, le GRU gagne dès 11 000 paramètres.
- Agrandir k au-delà de 30 ne sert jamais : la courbe du cisaillé remonte, comme en 5.4.

## 5.8 Sur des systèmes physiques réels

Les trois systèmes de la partie 8.3 (simulation, 500 neurones, 3 graines) ; RMSE de test.

| RMSE de test | Silverbox, multisinus | Silverbox, flèche avec extrapolation | Silverbox, flèche sans extrapolation | Wiener-Hammerstein |
|---|---|---|---|---|
| Réservoir accordé (1 500 par.) | 4,25 mV | 9,93 mV | 4,03 mV | 11,8 mV |
| GRU (754 500 par.) | 4,95 mV | **7,64 mV** | 4,08 mV | **3,17 mV** |
| Cisaillé en bande, k = 1 (1 499 par.) | 4,90 mV | 10,3 mV | 4,39 mV | 12,8 mV |
| Cisaillé en bande, k = 3 (2 494 par.) | 3,64 mV | 9,10 mV | 3,56 mV | 9,82 mV |
| **Cisaillé en bande, k = 10** (5 945 par.) | **2,93 mV** | 8,13 mV | **2,79 mV** | 7,84 mV |
| Cisaillé en bande, k = 10, avec amplitudes (6 445 par.) | **2,85 mV** | 8,12 mV | **2,73 mV** | 7,64 mV |
| Cisaillé à porte d'entrée, k = 10 (6 945 par.) | **2,92 mV** | 8,05 mV | **2,76 mV** | 7,75 mV |
| Cisaillé en bande, k = 30 (15 535 par.) | 3,93 mV | 8,71 mV | 3,55 mV | 9,42 mV |
| Cisaillé en bande, k = 100 (45 950 par.) | 6,78 mV | 12,9 mV | 8,16 mV | 13,1 mV |
| Cisaillé plein (125 750 par.) | 10,8 mV | 15,4 mV | 10,4 mV | 18,3 mV |

- **Sur Silverbox, k = 10 fait mieux que le GRU sur deux tests sur trois** : 2,9 mV contre 4,9 mV sur le multisinus (1,7 fois),
  2,8 contre 4,1 mV sans extrapolation, avec 127 fois moins de paramètres. Le GRU garde l'avantage quand il faut extrapoler
  (7,6 contre 8,1 mV).
- **Sur Wiener-Hammerstein**, k = 10 fait 1,5 fois mieux que l'accordé, mais le GRU reste 2,5 fois devant.
- Sur ces systèmes, la même courbe en k qu'en 5.4 : un optimum vers k = 10, puis une dégradation.
- **Les amplitudes n'apportent presque rien** au cisaillé (2,85 contre 2,93 mV), alors qu'elles sont indispensables à
  l'accordé sur ces mêmes systèmes (partie 4.2).
- **Sur Cascaded Tanks** (1 024 points d'entraînement), aucun modèle n'est utilisable : NMSE de 0,6 au mieux (accordé, k = 1),
  la régression linéaire reste devant tous (partie 8.3).

## 5.9 Face aux modèles existants, avec le même budget de recherche

Les quatre modèles cisaillés ont reçu le budget de recherche de la partie 8.2 : configuration par défaut + 12 essais jugés sur
validation (8 sur Lorenz et Hénon, pour tous les modèles), **k compris dans la recherche** (1, 3, 10, 30 ou 100), puis 5
graines mesurées une fois sur le test. Facteur d'erreur par rapport au meilleur modèle existant réglé (GRU ou LRU, taille
cherchée comprise) :

| Tâche | Meilleur existant | Cisaillé en bande | Cisaillé + amplitudes | Cisaillé à porte d'entrée | Cisaillé linéaire à porte d'entrée |
|---|---|---|---|---|---|
| Mackey-Glass | GRU (754 500 par.), 1.25e-7 | **×3,0** (k = 10) | **×3,3** (k = 10) | **×2,0** | ÷19 |
| NARMA10 | LRU (251 750 par.), 0.0137 | **×2,1** (k = 10) | **×1,9** (k = 1) | **×3,6** | **×8,3** |
| Mémoire | LRU (161 000 par.), 1.0e-4 | **÷7,2** | **÷9,1** | **÷5,8** | **×23** |
| Hénon | GRU (5 160 par.), 0.0015 | ÷1,5 (non significatif) | ÷2,5 (non significatif) | **÷5,2** | **÷29** |
| Lorenz | GRU (754 500 par.), 2.4e-7 | **÷16** | **÷22** | **÷37** | **÷780** |

- **Mackey-Glass et NARMA10** : avec le même budget de recherche, le cisaillé bat le GRU et le LRU réglés, avec 40 à 130 fois
  moins de paramètres. L'accordé, qui faisait déjà mieux qu'eux (partie 8.2), est dépassé de 1,7 fois sur Mackey-Glass et de
  1,2 fois sur NARMA10.
- **La mémoire** reste au LRU, sauf pour la variante linéaire à porte d'entrée, 23 fois meilleure que lui : une récurrence
  linéaire, une porte qui ne dépend que de l'entrée, une lecture GELU, comme dans les RNN linéaires à portes (minGRU, LRU).
- **Lorenz et Hénon** restent au GRU.
- k retenu par la recherche (cisaillé en bande) : 10 sur Mackey-Glass et NARMA10, 3 sur la mémoire et Lorenz, 30 sur Hénon ;
  toujours dans le plateau de la section 5.4.

## 5.10 Le cisaillé à portes : économiser des poids d'un GRU

Une cellule de type GRU a trois blocs de poids (porte de mise à jour z, porte de réinitialisation r, candidat), chacun avec une
matrice récurrente N×N. Chaque bloc peut être appris entier, cisaillé en bande (k = 10) sur un tirage aléatoire figé, ou figé ;
on peut aussi restreindre ce qui est entraîné. 200 neurones, cinq tâches, 3 graines ; référence : le GRU de 200 unités
(121 800 paramètres), avec les mêmes graines.

![Cisaillé à portes](../results/figures/shear_gated.png)

| Variante (paramètres) | Mackey-Glass | NARMA10 | Mémoire | Lorenz | Hénon |
|---|---|---|---|---|---|
| GRU, 200 unités (121 800), NMSE | 9.0e-7 | 0.132 | 0.63 | 3.6e-7 | 0.0036 |
| Portes z et r apprises, candidat cisaillé, −33 % (83 145) | ÷1,7 | **×8,9** | **×530** | **÷130** | **×1,5** |
| Porte z apprise, r et candidat cisaillés, −66 % (45 090) | ÷1,3 | **×10** | **×470** | **÷200** | **÷1,9** |
| Portes cisaillées, candidat appris entier, −66 % (45 090) | **÷47** | **×1,7** | **×36** | **÷79** | **÷35** |
| Portes et candidat cisaillés, −99 % (7 035) | ÷2,0 | **×20** | **×470** | **÷47** | **÷3,0** |
| Porte d'entrée (type minGRU), candidat cisaillé (2 745) | **÷2,1** | **×18** | **×460** | **÷68** | **÷4,9** |
| Récurrence linéaire à porte d'entrée (2 745) | **÷89** | **×59** | **×11 000** | **÷1 400** | **÷30** |
| Portes apprises, candidat figé, −33 % (81 200) | ÷1,3 | **×6,9** | **×140** | **÷430** | ×1,3 |
| Entraîné : entrées seules (800) | **÷4,4** | **×3,5** | **×52** | **÷78** | **÷5,8** |
| Entraîné : portes seules (4 290) | **÷30** | **×1,3** | **×1,6** | **÷1 500** | **÷130** |
| Entraîné : récurrence seule (1 945) | **÷20** | **÷2,0** | **×38** | **÷560** | **÷250** |
| Entraîné : entrées et récurrence (2 745) | **÷2,0** | **×18** | **×470** | **÷54** | **÷3,4** |
| *LRU, 200 unités (40 700)* | **÷420** | **×10** | **×110** | **÷330 000** | **÷5,8** |
| *Accordé, 200 neurones (600)* | **÷4,0** | **×8,5** | **×350** | **÷540** | **÷16** |

- **Économiser 33 % ou 66 % des poids** : la variante qui garde les portes apprises et cisaille le candidat égale le GRU sur
  Mackey-Glass et le bat sur Hénon (1,5 fois), NARMA10 et la mémoire ; elle perd 130 fois sur Lorenz. Cisailler aussi la
  porte r (−66 %) garde Mackey-Glass, perd Hénon.
- **À 99 % de poids en moins**, les cellules cisaillées battent le GRU sur NARMA10 et la mémoire, tâches où ce GRU échoue, et
  perdent sur les trois autres.
- **Ce qui est entraîné compte** : les entrées seules, les portes seules ou la récurrence seule ne suffisent pas ; les
  entrées avec la récurrence retrouvent la cellule complète.
- **La récurrence linéaire à porte d'entrée** est le meilleur modèle mesuré sur NARMA10 et la mémoire (100 fois mieux que le
  LRU de 200 unités sur la mémoire), et l'un des pires sur les tâches chaotiques.
- Aucune variante ne bat le GRU sur Lorenz : le chaos fait de produits reste son domaine.

## 5.11 Bilan de la partie 5

**Où ça marche.**
- En bande (k = 3 à 10), partout où l'accordé marche, et mieux : à 500 neurones, sur les cinq tâches synthétiques, avec un
  seul k pour toutes, et sur des séries régénérées.
- Avec le même budget de recherche, mieux que le GRU sur Mackey-Glass et que le LRU sur NARMA10.
- Sur Silverbox, mieux que le GRU sur deux tests sur trois, avec 127 fois moins de paramètres.
- La variante linéaire à porte d'entrée gagne nettement sur NARMA10 et la mémoire.

**Les limites.**
- **Le cisaillement n'explique pas le gain** (section 5.5) : le même nombre de coefficients appris au hasard dans le
  triangle, ou directement dans `W₀`, fait aussi bien.
- Le chaos fait de produits (Lorenz, Hénon) reste au GRU, même à budget égal au-delà de 100 000 paramètres.
- Wiener-Hammerstein reste au GRU (2,5 fois), et deux systèmes réels (Cascaded Tanks, EMPS) font échouer tous les modèles.
- Au-delà de k = 20, sur-apprentissage ; 3 graines dans la plupart des mesures (5 dans la recherche équitable).

**Le lien avec la théorie.**
- *Le cadre K·A·N.* Les parties 4 à 6 parcourent trois décompositions classiques : `K·A·K` (accordé), `K·A·N` (cisaillé),
  Jordan-Chevalley (nilpotent). Chacune sépare une pièce que les autres ne peuvent pas apprendre seule : le cisaillé change
  les valeurs propres et les valeurs singulières, et les directions singulières de façon contrainte, ce que ni l'accordé ni le
  tourné ne font. Ce cadre a permis de trouver le modèle, mais les contrôles montrent qu'il ne décrit pas ce qui le fait
  marcher.
- *Non-normalité.* La lecture de départ était celle de l'amplification transitoire : une matrice non normale peut amplifier
  une perturbation pendant quelques pas avant qu'elle ne s'éteigne (Trefethen et Embree, *Spectra and Pseudospectra*, 2005),
  ce qui allonge la mémoire ([Ganguli, Huh et Sompolinsky, 2008](https://www.pnas.org/doi/10.1073/pnas.0804451105)), et la
  bande dans la base `Q` ressemble à une chaîne d'anticipation. Les contrôles ne la soutiennent pas : la bande ne compte pas,
  et une correction creuse dans la base des neurones fait aussi bien. Toute correction de `W₀` modifie sa non-normalité ; rien
  ici ne montre que c'est ce qui compte.
- *Sur-apprentissage et nombre de paramètres.* Le plateau de k = 2 à 10 puis la remontée de l'erreur de test, avec une erreur
  d'entraînement qui baisse, est la courbe classique de la capacité : quelques milliers de coefficients récurrents suffisent
  sur des séries de 3 000 points ; davantage sur-apprend.
- *Travaux voisins.* Le nnRNN ([Kerg et al., 2019](https://arxiv.org/abs/1905.12080)) écrit `W = Q·(Λ + T)·Qᵀ` (forme de Schur,
  une similitude : les valeurs propres restent celles de `Λ`) et apprend tout, `Q` compris ; le cisaillé est un produit à
  gauche, `Q·(R + S)`, dont la diagonale `a` n'est pas le spectre, avec `Q` figée. Aucun travail trouvé n'apprend le facteur
  N de la décomposition d'Iwasawa d'un réservoir, mais les contrôles en font une piste écartée plutôt qu'un résultat. Apprendre
  une partie creuse des poids d'un réseau aléatoire est connu ailleurs (réglage fin creux, supermasks, partie 2) ; les RNN
  linéaires à portes (minGRU, LRU) sont les voisins de la variante linéaire.

---

[Sommaire](../README.md#le-rapport) · [← Partie 4](4_accorde.md) · [Partie 6 →](6_nilpotent.md)
