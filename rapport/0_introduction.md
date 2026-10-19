[Sommaire](../README.md#le-rapport) · [Partie 1 →](1_melanger.md)

# Introduction — entre un réservoir et un RNN

**La question.** Un réservoir (Echo State Network) tire sa matrice récurrente au hasard, la fige, et n'apprend que sa
lecture. Un RNN apprend tout. Que se passe-t-il entre les deux, quand on laisse la récurrence apprendre **un peu**, et
d'une façon choisie ? Chaque partie de ce rapport fixe une règle (« on n'apprend qu'une vitesse », « on n'apprend qu'une
atténuation par connexion », « on n'apprend qu'une amplitude par direction »…), mesure ce qu'elle vaut, dit sur quelles
tâches elle marche, et la relie à ce que la théorie et la littérature en disent déjà.

![Deux extrêmes, et ce qu'on étudie entre les deux](../results/figures/schema_principe.png)

![Entre un réservoir et un RNN : chaque partie choisit ce que la récurrence a le droit d'apprendre](../results/figures/schema_continuum.png)

## Le plan

| Partie | Ce que la récurrence a le droit d'apprendre | Fichier |
|---|---|---|
| 1 | tout, mais lentement : **mélanger** réservoir et RNN revient à **freiner** la récurrence | [1_melanger.md](1_melanger.md) |
| 2 | une atténuation entre 0 et 1 par connexion : le **réservoir sculpté** | [2_sculpte.md](2_sculpte.md) |
| 3 | une rotation, sans toucher au spectre : le **réservoir tourné** | [3_tourne.md](3_tourne.md) |
| 4 | une amplitude par direction singulière : le **réservoir accordé**, et toutes ses extensions | [4_accorde.md](4_accorde.md) |
| 5 | un cisaillement triangulaire (décomposition d'Iwasawa) : le **réservoir cisaillé** | [5_cisaille.md](5_cisaille.md) |
| 6 | les modes propres et une partie nilpotente (Jordan-Chevalley) : le **réservoir nilpotent** | [6_nilpotent.md](6_nilpotent.md) |
| 7 | rien dans le réservoir : un réservoir figé et un RNN distincts, **couplés** | [7_couplage.md](7_couplage.md) |
| 8 | **synthèse** : toutes les familles comparées, données non synthétiques, modèles existants, littérature | [8_synthese.md](8_synthese.md) |
| 9 | le choix lui-même : **la tâche peut-elle choisir sa structure ?** à partir du gradient, et sur des tâches où la structure est plantée | [9_structure.md](9_structure.md) |

Chaque partie sur un réservoir se termine par le même bilan en trois points : **où ça marche**, **les limites**, et
**le lien avec la théorie**. Les pistes encore ouvertes et la règle d'arrêt sont rassemblées dans l'[annexe](annexe_pistes.md).

## Les notations

Un réseau récurrent met à jour son état ainsi :

```
x(t+1) = tanh( W_in·u(t) + W·x(t) + b ),      ŷ(t) = W_out·x(t) + c
```

- Un **RNN classique** apprend `W_in`, `W`, `b` et `W_out` par rétropropagation à travers le temps.
- Un **réservoir classique** tire `W_in` et `W` au hasard, les fige, et n'apprend que `W_out` (par régression ridge).

Dans tout le rapport, **`W₀`** est la matrice aléatoire de départ (toujours figée) et **`ΔW`** ce qui est appris en plus.
Sauf mention contraire, `W_in` et `b` apprennent aussi : ce point compte beaucoup, et la partie 4 montre qu'une grande
part du gain vient d'eux.

| Nom | Matrice récurrente | Ce qui est appris dans la récurrence | Partie |
|---|---|---|---|
| **Réservoir classique** | `W = W₀` | rien | — |
| **RNN classique** | `W` entière (départ `W₀`) | tout | — |
| **RNN ralenti** × f | idem, tout apprend f fois moins vite | tout | 1 |
| **RNN freiné** | idem, seule la récurrence apprend 10 fois moins vite | tout | 1 |
| **Mélange fixe** | `W = (1−α)·W₀ + α·ΔW`, α fixé | `ΔW` | 1 |
| **Mélange réglé** | idem, un `α_rec` pour `W` et un `α_in` pour `W_in`, appris | `ΔW`, `α_in`, `α_rec` | 1 |
| **Réservoir retouché** | `W = W₀ + M ⊙ ΔW`, `M` = les connexions existantes | ces connexions | 1 |
| **Réservoir corrigé de rang r** | `W = W₀ + U·Vᵀ` | `U`, `V` (rang r) | 1 |
| **Réservoir sculpté** | `W = m ⊙ W₀`, `m ∈ [0, 1]` par connexion | `m` | 2 |
| **Réservoir sculpté intégral** | idem, et `W_in = m' ⊙ W_in₀` | les masques seulement | 2 |
| **Réservoir élagué** (contrôle) | `W = m ⊙ W₀`, `m ∈ {0, 1}` | quelles connexions garder | 2 |
| **Réservoir repondéré** (contrôle) | `W = g ⊙ W₀`, `g` libre | `g` | 2 |
| **Réservoir tourné** | `W = Q·W₀·Qᵀ`, `Q` orthogonale | `Q` | 3 |
| **Réservoir accordé** | `W = U·diag(σ ⊙ g)·Vᵀ` (décomposition de `W₀`) | `g`, une amplitude par direction | 4 |
| **Réservoir à biais et entrées appris** (contrôle « sans g ») | `W = W₀` | rien ; seuls `W_in` et `b` apprennent | 4 |
| **Réservoir cisaillé** | `W = Q·(R + S)` (décomposition QR de `W₀`) | `S`, partie strictement triangulaire | 5 |
| **Réservoir cisaillé en bande** | idem, `S` limitée aux k premières sur-diagonales | k·N coefficients de `S` | 5 |
| **Réservoir nilpotent** | `W = S(g, φ) + Qr·T·Qrᵀ` | modes propres et partie nilpotente | 6 |
| **Duo croisé, relais aléatoire, relais appris** | deux réseaux distincts | le RNN et ses liens au réservoir | 7 |

Pour voir ce que chaque modèle fait au réservoir, voici chacun entraîné sur un petit réseau de 20 neurones (même réservoir
de départ, même tâche). À gauche, le réseau après entraînement ; à droite, ce qui a réellement changé.

![Galerie des modèles](../results/figures/model_gallery.png)

Le RNN classique **crée** des connexions partout ; le RNN freiné et le mélange fixe font les mêmes changements, en beaucoup
plus faible. Le réservoir retouché ne touche qu'aux connexions existantes. Le réservoir sculpté ne peut que les affaiblir
ou les couper ; le repondéré peut aussi les renforcer ou les inverser. Le réservoir tourné change tout le câblage mais garde
exactement le même spectre.

![Combien chaque modèle apprend](../results/figures/schema_budget.png)

## Le protocole

- **Tâches principales** : séries de 4 000 pas, 3 000 pour l'entraînement et 1 000 pour le test.
  *Mackey-Glass* : série chaotique (τ = 17), prédire 15 pas à l'avance.
  *NARMA10* : système non linéaire à mémoire d'ordre 10, banc d'essai classique des réservoirs.
  *Mémoire* : rappeler l'entrée d'il y a 15 pas (entrée aléatoire uniforme).
  *Lorenz* (prédire x à +10 pas) et *Hénon* (+3 itérations) : deux systèmes chaotiques faits de produits de variables,
  ajoutés pour les parties 3 à 7.
- **Réseau** : 100 neurones (500 pour les comparaisons de grande taille), réservoir creux (10 % de connexions) de rayon
  spectral 0.9, échelle d'entrée 0.5. Partie 2 : réservoir « dense » en plus (toutes les connexions, même rayon spectral).
- **Entraînement** : rétropropagation à travers le temps (numpy à 100 neurones, PyTorch au-delà ; les deux versions sont
  vérifiées identiques), Adam (lr 2e-3), 1 500 pas, lots de 32 fenêtres de 60 pas, gradient écrêté à 1, readout de départ
  nul.
- **Évaluation** : le readout final est recalculé par régression ridge sur les états, puis on mesure la NMSE
  (`MSE / variance de la cible`) sur le test.
- **Graines appariées** : pour une graine donnée, tous les modèles partent du même réservoir et voient les mêmes fenêtres ;
  les comparaisons utilisent l'écart apparié `log10(NMSE / NMSE_réf)` (−0,3 ≈ deux fois moins d'erreur). Un écart compte
  comme réel s'il dépasse max(2 erreurs types ; 0,05).
- **Vérifications** : gradients contrôlés par différences finies, version PyTorch identique à la version numpy, et un test
  par modèle (le dépôt en contient plus de 200).

---

[Sommaire](../README.md#le-rapport) · [Partie 1 →](1_melanger.md)
