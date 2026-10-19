"""
Partie 4.2 — Pourquoi le réservoir accordé marche-t-il ? Trois contrôles, à 500 neurones (GPU conseillé).

    Directions    : directions aléatoires de W₀ (accordé) ou directions de Fourier structurées (accordé rapide,
                    même nombre d'amplitudes apprises, voir partie 7) ; et ces réservoirs sans entraînement.
    Spectre       : mêmes directions aléatoires, mais spectre de départ plat (réservoir orthogonal) ou très étalé.
    Nombre        : seules les k plus grandes directions sont accordées (k = N/50, N/10, N/5, N/2).

Même protocole et même code que la partie 4.2 (experiments/15_improvements.py).

    python experiments/18_spectrum_controls.py             # N = 500, 3 seeds, 3 tâches (~1 h 30 sur un T4)
    python experiments/18_spectrum_controls.py --report
"""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("improvements", Path(__file__).with_name("15_improvements.py"))
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
LR = E.LR


def models(n):
    ks = [max(1, n // 50), n // 10, n // 5, n // 2]
    return ([
        ("réservoir classique", {"kind": "reservoir"}, "gray"),
        ("RNN, lr global × 0.1", {"kind": "convex", "alpha": 1.0, "lr": 0.1 * LR}, "tab:red"),
        ("RNN, lr global × 0.03", {"kind": "convex", "alpha": 1.0, "lr": 0.03 * LR}, "salmon"),
        ("réservoir accordé", {"kind": "spectrum"}, "tab:blue"),
        # directions
        ("réservoir de Fourier, non entraîné", {"kind": "fast_reservoir"}, "silver"),
        ("accordé, directions de Fourier", {"kind": "fast_tuned"}, "tab:green"),
        # spectre de départ
        ("accordé, spectre de départ plat", {"kind": "tuned", "spectrum": "flat"}, "tab:orange"),
        ("accordé, spectre de départ étalé", {"kind": "tuned", "spectrum": "spread"}, "tab:orange"),
    ] + [(f"accordé, {k} directions sur {n}", {"kind": "tuned", "topk": k}, "tab:cyan") for k in ks])


def main():
    n = 500
    for i, a in enumerate(sys.argv):
        if a == "--n" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
    E.MODELS = models(n)
    E.NEW = [l for l, m, _ in E.MODELS if l not in ("réservoir classique", "réservoir accordé") and
             not l.startswith("RNN")]
    E.RNNS = ["RNN, lr global × 0.1", "RNN, lr global × 0.03"]
    E.PREFIX = "spectrum_controls"
    E.DEFAULT_N = 500
    E.TITLE = ("Partie 3 — pourquoi le réservoir accordé marche : directions, spectre de départ, nombre de directions. "
               "Tirets rouges : meilleur RNN ; pointillés bleus : réservoir accordé.")
    E.main()


if __name__ == "__main__":
    main()
