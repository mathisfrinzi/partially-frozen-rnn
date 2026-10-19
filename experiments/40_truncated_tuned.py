"""
Partie 4.2 — Accordé tronqué : quelles directions faut-il accorder quand on n'en accorde que quelques-unes ?

L'expérience 18 (partie 2.6) a déjà accordé les k plus grandes directions (k = 10, 50, 100, 250 sur 500). Elle ne dit pas
si « les plus grandes » est le bon choix. On compare ici, à budget r égal (r = 2, 10, 50 amplitudes apprises sur 500) :

    plus grandes   les r plus grandes valeurs singulières de W₀        (select="top")
    plus petites   les r plus petites                                  (select="bottom")
    au hasard      r directions tirées au hasard                       (select="random")

Les autres directions gardent g = 1 (réservoir d'origine). W_in, les biais et le readout sont appris dans tous les cas.

Même protocole et même code que 15_improvements.py (1 500 pas, readout ridge, 3 tâches).

    python experiments/40_truncated_tuned.py              # N = 500, 3 seeds (Colab, ~1 h 30 sur un T4)
    python experiments/40_truncated_tuned.py --report
"""
import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("improvements", Path(__file__).with_name("15_improvements.py"))
E = importlib.util.module_from_spec(spec)
spec.loader.exec_module(E)
LR = E.LR
RS = (2, 10, 50)
SELECTS = (("top", "plus grandes", "tab:blue"), ("bottom", "plus petites", "tab:red"), ("random", "au hasard", "tab:gray"))


def models(n):
    rs = [r for r in RS if r < n]
    return ([
        ("réservoir classique", {"kind": "reservoir"}, "gray"),
        ("RNN, lr global × 0.1", {"kind": "convex", "alpha": 1.0, "lr": 0.1 * LR}, "tab:red"),
        ("RNN, lr global × 0.03", {"kind": "convex", "alpha": 1.0, "lr": 0.03 * LR}, "salmon"),
        ("réservoir accordé", {"kind": "spectrum"}, "tab:blue"),
    ] + [(f"accordé, {r} {name}", {"kind": "tuned", "topk": r, "select": sel}, color)
         for r in rs for sel, name, color in SELECTS])


def main():
    n = 500
    for i, a in enumerate(sys.argv):
        if a == "--n" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
    E.MODELS = models(n)
    E.NEW = [l for l, m, _ in E.MODELS if l.startswith("accordé,")]
    E.RNNS = ["RNN, lr global × 0.1", "RNN, lr global × 0.03"]
    E.PREFIX = "truncated_tuned"
    E.DEFAULT_N = 500
    E.TITLE = ("Partie 2.9 — accordé tronqué : r directions accordées, choisies parmi les plus grandes, les plus "
               "petites ou au hasard. Tirets rouges : meilleur RNN ; pointillés bleus : réservoir accordé.")
    E.main()


if __name__ == "__main__":
    main()
