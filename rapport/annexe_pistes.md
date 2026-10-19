[Sommaire](../README.md#le-rapport) · [← Partie 9](9_structure.md)

# Annexe — Dernières mesures, pistes ouvertes et règle d'arrêt

## Les quatre dernières mesures

| Mesure | Question | Réponse | Partie |
|---|---|---|---|
| Biais + `W_in` + rang r, sans g, sur Lorenz et Hénon | retirer g améliore-t-il l'accordé de rang r là où g nuit ? | **oui**, un peu (1,3 à 2,2 fois) ; 16 à 17 fois mieux que le GRU de même taille jusqu'à 33 000 paramètres | 4.9 |
| Systèmes réels : sans g, cisaillé plein et en bande | les biais et entrées suffisent-ils sur Silverbox ? le cisaillement survit-il au réel ? | **non** pour sans g (1,7 à 3 fois pire) ; **oui** pour le cisaillé en bande (meilleur de nos modèles sur Silverbox, à égalité avec le GRU) ; non pour le cisaillé plein | 4.2, 5.8, 8.3 |
| Recherche équitable avec le réservoir à biais et entrées appris | le modèle le plus simple tient-il face aux modèles existants ? | **oui** sur Mackey-Glass et la mémoire (égal à l'accordé), non sur NARMA10 (2 fois moins bien) | 4.2, 8.2 |
| Cisaillé et nilpotent à 500 neurones, pleins et en bande | la non-normalité tient-elle à grande taille, et à faible coût ? | **oui en bande** : le cisaillé k = 10 bat l'accordé sur les cinq tâches ; les versions pleines s'effondrent | 5.3, 6.2 |

## La confirmation du cisaillé en bande

Le cisaillé en bande était le seul résultat positif obtenu après coup. Neuf blocs du [carnet Colab](../notebooks/colab_gpu.ipynb)
(`experiments/43_shear_study.py` et `26_fair_search.py`) l'ont mesuré :

| Mesure | Question | Réponse | Partie |
|---|---|---|---|
| Courbe en k | le gain tient-il à un k réglé tâche par tâche ? | **non** : plateau de k = 2 à 10, un seul k = 10 fait presque aussi bien que le meilleur k de chaque tâche ; sur-apprentissage au-delà de k = 20 | 5.4 |
| Contrôles « positions au hasard », « dans `W₀` » | la bande compte-t-elle ? la décomposition compte-t-elle ? | **non et non** : le même nombre de coefficients, au hasard dans le triangle ou directement dans `W₀`, fait aussi bien | 5.5 |
| Données fraîches | le classement tient-il sur des séries jamais vues ? | **oui** contre l'accordé ; le GRU reste devant sur Lorenz et Hénon | 5.6 |
| EMPS | sur un système réel jamais utilisé ? | aucun modèle n'y fonctionne, GRU compris | 5.6 |
| Budget égal contre le GRU | où se place le cisaillé sur la courbe erreur / paramètres ? | devant le GRU sur Mackey-Glass, NARMA10, la mémoire ; derrière au-delà de 100 000 paramètres sur Lorenz et Hénon ; derrière dès 11 000 sur Wiener-Hammerstein | 5.7 |
| Systèmes réels, courbe en k | avec amplitudes ? | k = 10 fait mieux que le GRU sur deux tests de Silverbox ; les amplitudes n'apportent presque rien | 5.8 |
| Recherche équitable | face au GRU et au LRU réglés ? | mieux sur Mackey-Glass (3 fois) et NARMA10 (2 fois) ; la variante linéaire à porte d'entrée 23 fois mieux sur la mémoire ; le GRU gagne sur Lorenz et Hénon | 5.9, 8.2 |
| Cisaillé à portes | peut-on économiser 33 %, 66 % ou 99 % des poids d'un GRU ? | oui sur NARMA10, la mémoire et Hénon (−33 %) ; jamais sur Lorenz | 5.10 |

## Pistes ouvertes, non codées

- **Une base de plasticité intrinsèque** à côté de l'accordé et du réservoir à biais et entrées appris, avec la règle exacte
  de Schrauwen et al. (2008).
- **Le sculpté intégral dans la recherche équitable**, sur la mémoire, où il est le meilleur de nos modèles et où le LRU
  bien réglé est l'adversaire.
- **Un protocole long** (10 000 pas, taille choisie sur validation) pour l'accordé et le GRU sur Silverbox et
  Wiener-Hammerstein, pour se comparer au GRU publié dans ses conditions.
- **Un test explicite de μP** sur des RNN issus d'un réservoir : le learning rate récurrent optimal varie-t-il comme 1/N, et
  se transfère-t-il d'une taille à l'autre ?
- **NVAR complet** comme base dans la carte des tâches (le readout carré n'en est qu'une version minimale) ; un mélange
  additif de même taille comme référence pour les duos croisés (partie 7).
- **Un filtre polynomial de la matrice**, `W = Σₖ cₖ W₀ᵏ` (K coefficients) : seul candidat sans voisin trouvé, à vérifier dans la
  littérature (*matrix polynomial*, *graph filter*, *Krylov*) avant de coder.
- **Une explication** de ce que les biais appris font à la dynamique (point de fonctionnement de chaque neurone, pente locale
  du `tanh`), et de pourquoi g allonge la mémoire : un résultat simple vaudrait plus que trois tâches de plus.
- **Écartés après vérification de la littérature**, à citer mais à ne pas coder comme modèles nouveaux : la forme de Schur
  (nnRNN), la partie symétrique/antisymétrique (Euler State Networks, RNN antisymétriques), les rotations de Givens (QGOFT,
  DizzyRNN), le taux de fuite appris (réservoir à fuite), un accordé à portes dont une des trois matrices serait entièrement
  apprise (il revient à un GRU partiellement contraint, sans idée nouvelle).

## Règle d'arrêt

Trois questions décidaient de la suite :

1. Un relais combiné à l'accordé garde-t-il le gain du relais sur Hénon (partie 7) ? **Non.**
2. Le readout carré ou le relais réduisent-ils de moitié l'écart au GRU sur Wiener-Hammerstein ? **Non, définitivement.**
3. La non-normalité garde-t-elle son gain sur au moins trois tâches sur cinq ? **Le gain, oui ; la non-normalité, non** : le
   cisaillé en bande gagne sur les cinq tâches synthétiques et se confirme (section précédente), mais les contrôles montrent
   que le même nombre de coefficients appris ailleurs fait aussi bien.

Les trois questions sont tranchées. La suite était la partie 9 : la tâche peut-elle choisir sa structure ? La réponse est
non, pas à partir du gradient, et la partie 9 la tranche aussi sur des tâches où la structure est plantée.


---

[Sommaire](../README.md#le-rapport) · [← Partie 9](9_structure.md)
