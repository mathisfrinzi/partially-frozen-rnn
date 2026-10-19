"""
Parties 4.5 et 4.6 — Réservoir accordé à portes : le chemin RNN → GRU / LSTM, appliqué au réservoir accordé.

Chaque matrice récurrente d'une cellule à portes devient une matrice accordée, U·diag(σ ⊙ g_k)·Vᵀ, avec les
directions aléatoires figées du réservoir (voir src/hfrnn/torch_tuned.py, GatedTuned) :

    réservoir accordé à portes                portes de type GRU (z, r) à matrices accordées (≈ 9 N paramètres)
    réservoir accordé à portes légères        portes neurone par neurone (≈ 9 N)
    réservoir accordé à portes, type LSTM     portes f, i, o à matrices accordées (≈ 12 N)
    réservoir accordé à porte d'entrée        une seule porte, qui ne dépend que de l'entrée, comme minGRU (≈ 5 N)
    réservoir accordé linéaire (type LRU)     récurrence linéaire diagonale complexe ; seules les valeurs propres,
                                              la normalisation et les entrées apprennent (≈ 2,5 N ; LRU complet ≈ N²)

Toutes partent (presque) du réservoir accordé. Références : réservoir accordé, accordé sélectif et accordé + fuite
(partie 4.2), RNN, GRU et LSTM bien initialisés (partie 6).

    python experiments/21_gated.py --n 200 --seeds 2       # test rapide (CPU, ~1 h)
    python experiments/21_gated.py                          # N = 500, 3 seeds (GPU, ~2 h)
    python experiments/21_gated.py --report
"""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("improvements", Path(__file__).with_name("15_improvements.py"))
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
LR = E.LR

GATED = [
    ("réservoir accordé à portes", {"kind": "gated_tuned", "cell": "gru"}, "tab:cyan"),
    ("réservoir accordé à portes légères", {"kind": "gated_tuned", "cell": "gru_light"}, "tab:cyan"),
    ("réservoir accordé à portes, type LSTM", {"kind": "gated_tuned", "cell": "lstm"}, "tab:cyan"),
    ("réservoir accordé à porte d'entrée", {"kind": "gated_tuned", "cell": "mingru"}, "tab:cyan"),
]
# partie 5.4 : récurrence linéaire diagonale (type LRU) dont seules les valeurs propres et les entrées apprennent
LINEAR = [
    ("réservoir accordé linéaire (type LRU)", {"kind": "linear_tuned"}, "tab:green"),
    ("réservoir accordé linéaire, 2 couches", {"kind": "linear_tuned", "layers": 2}, "tab:green"),
]
LRU_REFS = [("LRU, lr 2e-3", {"kind": "lru"}, "tab:pink"), ("LRU, lr 5e-4", {"kind": "lru", "lr": 5e-4}, "tab:pink")]
# contrôle : seules les portes apprennent, la matrice principale reste le réservoir (esprit des Gated ESN)
CONTROL = [("contrôle : réservoir accordé à portes, portes seules", {"kind": "gated_tuned", "cell": "gru", "gates_only": True},
            "lightsteelblue")]


def models(quick):
    refs = [("réservoir accordé", {"kind": "spectrum"}, "tab:blue"),
            ("GRU, init soignée, lr 2e-3", {"kind": "gru", "rec_init": "ortho"}, "tab:orange")]
    if not quick:
        refs = [("réservoir classique", {"kind": "reservoir"}, "gray"),
                ("RNN, lr global × 0.1", {"kind": "convex", "alpha": 1.0, "lr": 0.1 * LR}, "tab:red"),
                ("RNN, lr global × 0.03", {"kind": "convex", "alpha": 1.0, "lr": 0.03 * LR}, "salmon"),
                ("GRU, init soignée, lr 2e-3", {"kind": "gru", "rec_init": "ortho"}, "tab:orange"),
                ("GRU, init soignée, lr 5e-4", {"kind": "gru", "rec_init": "ortho", "lr": 5e-4}, "tab:orange"),
                ("LSTM, init soignée, lr 5e-4", {"kind": "lstm", "rec_init": "ortho", "lr": 5e-4}, "goldenrod"),
                ("réservoir accordé", {"kind": "spectrum"}, "tab:blue"),
                ("accordé sélectif", {"kind": "tuned", "selective": True}, "lightsteelblue"),
                ("accordé + fuite", {"kind": "tuned", "leak": True}, "lightsteelblue")]
    return refs + GATED + ([] if quick else CONTROL + LRU_REFS + LINEAR)


def main():
    quick = any(a == "--n" for a in sys.argv) and "200" in sys.argv
    E.MODELS = models(quick)
    E.NEW = [l for l, _, _ in GATED + LINEAR]
    # référence « à battre » : le meilleur modèle existant récurrent testé (GRU bien initialisé, et RNN si présent)
    E.RNNS = ["GRU, init soignée, lr 2e-3"] if quick else ["GRU, init soignée, lr 2e-3", "GRU, init soignée, lr 5e-4",
                                                           "LSTM, init soignée, lr 5e-4", "RNN, lr global × 0.1",
                                                           "RNN, lr global × 0.03", "LRU, lr 2e-3", "LRU, lr 5e-4"]
    E.PREFIX = "gated"
    E.REF_LABEL = "meilleur modèle existant testé"
    E.DEFAULT_N = 500
    E.TITLE = ("Partie 5 — réservoir accordé à portes (cyan). Tirets rouges : meilleur modèle existant testé ; "
               "pointillés bleus : réservoir accordé.")
    E.main()


if __name__ == "__main__":
    main()
