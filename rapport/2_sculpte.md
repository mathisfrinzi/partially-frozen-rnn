[Sommaire](../README.md#le-rapport) · [← Partie 1](1_melanger.md) · [Partie 3 →](3_tourne.md)

# Partie 2 — Le réservoir sculpté : n'apprendre qu'une atténuation par connexion

![Réservoir sculpté](../results/figures/schema_sculpte.png)

> **Ce qu'il faut retenir**
>
> - Atténuer chaque connexion d'un réservoir dense par un coefficient entre 0 et 1, sans jamais changer la valeur d'un poids, bat le RNN : c'est le meilleur modèle de l'étude sur NARMA10 et la mémoire à 100 neurones, et le meilleur sur la mémoire à toutes les tailles testées.
> - C'est la **borne** qui compte : un masque binaire (élagué) ou un gain libre (repondéré) font nettement moins bien.
> - Le masque a autant de paramètres qu'un RNN dense : le gain porte sur ce que le réseau a le droit de faire, pas sur sa taille. L'idée est très proche des *supermasks*.

La partie 1 montre qu'il faut rester près du réservoir. Ici, on va plus loin : on ne touche plus du tout aux valeurs
aléatoires. Pour chaque connexion, on apprend seulement un coefficient `m ∈ [0, 1]` qui l'atténue (`W = m ⊙ W₀`, avec
`m = sigmoid(s)`, départ `m ≈ 0,95`). Une connexion peut être affaiblie ou coupée, jamais inversée ni amplifiée. Dans le
**réservoir sculpté intégral**, on fait de même pour les poids d'entrée : seuls ces masques, le biais et le readout
apprennent. Variante « réservoir dense » : toutes les connexions existent au départ (même rayon spectral 0,9), ce qui donne
beaucoup plus de matière à sculpter.

## 2.1 Atténuer des poids aléatoires suffit pour faire mieux qu'un RNN

100 neurones, 8 graines (5 pour la mémoire) :

| | Mackey-Glass | NARMA10 | Mémoire |
|---|---|---|---|
| Réservoir classique (creux / dense) | 0.0018 / 0.0016 | 0.279 / 0.260 | 0.785 / 0.782 |
| Meilleur RNN | 0.000085 | 0.0173 | 0.0027 |
| RNN, départ dense | 0.000071 | 0.0346 | 0.0028 |
| Partie 1 : RNN freiné | **0.000032** | 0.0082 | 0.0011 |
| Réservoir sculpté, départ creux (entrées libres) | 0.000048 | 0.0304 | 0.0035 |
| Réservoir sculpté, dense (entrées libres) | 0.000039 | 0.0084 | 0.0012 |
| **Réservoir sculpté intégral** (masques sur W et W_in, dense) | 0.000035 | **0.0078** | **0.00060** |
| Contrôle : réservoir repondéré (peut inverser ou amplifier) | 0.000038 | 0.0095 | 0.0017 |
| Réservoir élagué (on garde 50 % des connexions) | 0.00034 | 0.043 | 0.078 |

![Réservoir sculpté](../results/figures/supermask.png)

Le réservoir sculpté intégral bat le meilleur RNN sur les trois tâches (écart apparié −0,38 ± 0,08, −0,34 ± 0,05,
−0,65 ± 0,02 : **2,4, 2,2 et 4,5 fois moins d'erreur**). Il fait jeu égal avec le RNN freiné sur Mackey-Glass et NARMA10,
et 1,8 fois mieux sur la mémoire, alors qu'aucune valeur de poids n'est modifiée.

Le réservoir dense est indispensable : sur le réservoir creux (10 % de connexions), le sculpté fait au mieux jeu égal avec
le RNN. Un réservoir classique dense, lui, n'est pas meilleur qu'un creux : la densité ne sert qu'à donner plus de
connexions à moduler.

## 2.2 Ce n'est pas de l'élagage, c'est une atténuation fine

![Sculpté, élagué, repondéré](../results/figures/schema_sculpte_controles.png)

- **Réservoir élagué** (masque binaire : garder ou couper, comme les supermasks des réseaux d'images) : nettement moins
  bon que le RNN sur les trois tâches.
- **Binariser après coup** le masque continu appris (garder si `m > 0,5`) détruit la performance : 0,029 au lieu de 0,0006
  sur la mémoire, 0,033 au lieu de 0,0078 sur NARMA10. Pourtant, environ 80 % des connexions ont `m > 0,5` : le réseau garde
  presque tout, mais règle finement l'intensité de chaque connexion.

## 2.3 Ce qui compte : ne pas pouvoir inverser ni amplifier

Le **réservoir repondéré** est identique au sculpté, sauf que le coefficient n'est plus borné : chaque connexion peut
changer de signe ou être amplifiée. Il fait **2,9 fois moins bien sur la mémoire** (−0,46 ± 0,02), un peu moins bien sur
NARMA10 (−0,09 ± 0,02) et pareil sur Mackey-Glass. La contrainte elle-même aide. Réserve : un seul learning rate a été
testé pour ce contrôle, et diviser par 5 celui des masques change un peu les scores (mémoire 0,00081, NARMA10 0,0090).

## 2.4 À plus grande taille

- **500 neurones, 1 500 pas** (3 graines) : le sculpté intégral fait 13 fois mieux que le meilleur RNN sur Mackey-Glass et
  5 fois sur NARMA10 ; sur la mémoire, il fait **3 fois mieux que le réservoir accordé** (0,0006 contre 0,0019), et aucune
  variante de l'accordé ne l'égale.
- **500 neurones, 4 000 pas, face aux modèles existants** (réglages choisis sur le test) : sur la mémoire, 0,00018, soit le
  meilleur score de toute cette comparaison (27 fois devant le LRU dans ce protocole). Il n'a pas été inclus dans la
  recherche équitable de la partie 8.2, où un LRU bien réglé atteint 1,0e-4 : le sculpté y aurait un adversaire sérieux.
- **De 100 à 2 000 neurones** (3 graines) : premier sur la mémoire **à toutes les tailles** (5e-4 à 1e-3, 3 à 4,5 fois devant
  l'accordé) ; premier sur NARMA10 à 100 et 200 neurones, puis il se dégrade (0,017 à 500, 0,059 à 2 000 : le readout
  sur-apprend les 3 000 points) ; sur Mackey-Glass, premier à 100 et 1 000 neurones.

| Mémoire | N = 100 | 200 | 500 | 1 000 | 2 000 |
|---|---|---|---|---|---|
| Meilleur RNN | 0.016 | 0.015 | 0.036 | 0.044 | 0.044 |
| Réservoir accordé | 0.0029 | 0.0018 | 0.0019 | 0.0020 | 0.0044 |
| **Réservoir sculpté intégral** | **0.00065** | **0.00055** | **0.00060** | **0.00050** | **0.0010** |

- **Ce qu'il calcule** (capacité mémoire, 100 neurones) : entraîné sur NARMA10, c'est le modèle qui gagne le plus de capacité
  non linéaire (48 sur 100, contre 27 pour le RNN) ; les trois meilleurs modèles sur NARMA10 sont les trois qui en ont le plus.
- **Combiné à l'accordé**, il n'apporte rien de plus que seul : accordé + sculpté intégral fait exactement comme le sculpté
  intégral sur la mémoire (×3,1 contre ×3,1 à 500 neurones) et moins bien ailleurs.

## 2.5 Sur des données non synthétiques

- **MNIST ligne par ligne** (256 neurones, mêmes réglages pour tous) : le sculpté à entrées libres égale le RNN (4,48 %
  contre 4,41 %) sans changer la valeur d'aucun poids récurrent ; le sculpté intégral, qui n'apprend même pas les entrées,
  passe de 27,6 % (réservoir classique) à 6,4 % et progresse encore à la dernière époque.
- **Séries réelles** (100 neurones) : sur la température quotidienne de Melbourne, il fait mieux que le RNN (0,88 contre
  0,97, une NMSE de 1 étant la prévision naïve), mais moins bien que le réservoir classique (0,77), qui ne sur-apprend pas.

## 2.6 Bilan de la partie 2

**Où ça marche.** Les tâches de mémoire, à toutes les tailles (c'est le meilleur modèle de l'étude sur la mémoire), et
NARMA10 à petite taille. Il égale le RNN sur MNIST ligne par ligne.

**Les limites.**
- Le masque a N² paramètres : ce n'est pas un modèle économe. Il exige un réservoir dense.
- Il se dégrade quand le réseau grossit sur NARMA10, comme les autres modèles qui apprennent N² coefficients.
- Non mesuré sur Lorenz, Hénon et les systèmes physiques réels ; absent de la recherche équitable (partie 8.2). Le learning
  rate des masques et celui du repondéré n'ont pas été optimisés finement.

**Le lien avec la théorie.**
- *Une borne de stabilité.* Pour toute matrice `A`, le rayon spectral vérifie `ρ(A) ≤ ρ(|A|)`, et `ρ` est croissant en
  chaque coefficient d'une matrice positive (Perron-Frobenius). Avec `m ∈ [0, 1]`, on a donc
  `ρ(m ⊙ W₀) ≤ ρ(|m ⊙ W₀|) ≤ ρ(|W₀|)` : quoi qu'il apprenne, le réservoir sculpté ne peut pas dépasser le rayon spectral de
  `|W₀|`. Le repondéré, lui, n'a aucune borne. Cette propriété est compatible avec le fait que la borne aide surtout la
  mémoire (où une dynamique instable ou saturée coûte cher), sans en être une démonstration.
- *Masques sur des poids aléatoires.* Un sous-réseau performant se cache déjà dans un réseau aléatoire
  ([Ramanujan et al., 2020](https://arxiv.org/abs/1911.13299) ; Zhou et al., 2019), y compris dans des ResNets à poids partagés
  ([Hidden-Fold Networks](https://arxiv.org/abs/2111.12330)) ; des masques réels seuillés adaptent un réseau pré-entraîné
  ([Piggyback, Mallya et al., 2018](https://arxiv.org/abs/1801.06519)) ; n'entraîner que la normalisation
  ([Frankle et al., 2021](https://arxiv.org/abs/2003.00152)) ou que les biais
  ([Williams et al., ICLR 2025](https://arxiv.org/abs/2407.00957)) d'un réseau figé reste très expressif.
- *Côté réservoirs.* On trouve de l'élagage heuristique suivi d'un réapprentissage du readout
  ([Dynamical Mode Pruning, 2026](https://www.alphaxiv.org/abs/2608.04593)) et des portes qui modulent globalement l'entrée et le
  rayon spectral d'un réservoir figé ([Self-Modulated RC, 2023](https://arxiv.org/abs/2301.09235)) ; moduler le gain des neurones
  d'un réseau récurrent figé est étudié en neurosciences ([Stroud et al., 2018](https://cbl.eng.cam.ac.uk/hennequin/publications/stroud-natneuro-2018/)).
  Aucun travail trouvé n'apprend une atténuation bornée connexion par connexion sur un réservoir pour des séries temporelles,
  ni ne compare directement masque continu borné, masque binaire et gain libre ; mais la forme est identifiée.

---

[Sommaire](../README.md#le-rapport) · [← Partie 1](1_melanger.md) · [Partie 3 →](3_tourne.md)
