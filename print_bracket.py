#!/usr/bin/env python3
"""Print predicted WC 2026 bracket from knockout simulation output."""

import sys
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path(__file__).parent / "output"

R32 = [
    ("South Africa","Canada"),("Germany","Paraguay"),("Netherlands","Morocco"),
    ("Brazil","Japan"),("France","Sweden"),("Ivory Coast","Norway"),
    ("Mexico","Ecuador"),("England","DR Congo"),
    ("United States","Bosnia and Herzegovina"),("Belgium","Senegal"),
    ("Colombia","Ghana"),("Spain","Austria"),("Switzerland","Algeria"),
    ("Argentina","Cape Verde"),("Portugal","Croatia"),("Australia","Egypt"),
]
R16_PAIRS = [(1,4),(0,2),(3,5),(6,7),(14,10),(8,9),(13,15),(12,11)]
QF_PAIRS  = [(0,1),(4,5),(2,3),(6,7)]
SF_PAIRS  = [(0,1),(2,3)]

MATCH_NAMES = {
    "r32": [f"M{73+i}" for i in range(16)],
    "r16": [f"M{89+i}" for i in range(8)],
    "qf":  [f"M{97+i}" for i in range(4)],
    "sf":  ["M101","M102"],
    "final": ["Final"],
}


def fav(a, b, prob_col, probs):
    pa, pb = probs.loc[a, prob_col], probs.loc[b, prob_col]
    winner, loser = (a, b) if pa >= pb else (b, a)
    p = max(pa, pb) / (pa + pb) if (pa + pb) > 0 else 0.5
    return winner, loser, p


def main():
    files = sorted(OUTPUT_DIR.glob("knockout_probs_*.csv"))
    if not files:
        sys.exit("No knockout_probs_*.csv found in output/")
    csv = files[-1]
    df = pd.read_csv(csv).set_index("team")

    lines = [f"WC 2026 — Model Predicted Bracket (source: {csv.name})",
             "=" * 60, ""]

    def section(title, matchups, pairs, prev_winners, col):
        lines.append(f"{'— ' + title + ' —':^60}")
        lines.append("")
        winners = []
        for idx, (i, j) in enumerate(pairs):
            a, b = prev_winners[i], prev_winners[j]
            w, l, p = fav(a, b, col, df)
            mname = matchups[idx] if idx < len(matchups) else ""
            lines.append(f"  {mname:<5}  {w} ({p:.0%})  def. {l}")
            winners.append(w)
        lines.append("")
        return winners

    # R32
    lines.append("— Round of 32 —\n")
    r32_all = list(range(16))
    r32_teams = [t for pair in R32 for t in pair]
    r32_w = []
    for idx, (a, b) in enumerate(R32):
        w, l, p = fav(a, b, "R16", df)
        lines.append(f"  {MATCH_NAMES['r32'][idx]:<5}  {w} ({p:.0%})  def. {l}")
        r32_w.append(w)
    lines.append("")

    r16_w = section("Round of 16", MATCH_NAMES["r16"], R16_PAIRS, r32_w, "QF")
    qf_w  = section("Quarterfinals", MATCH_NAMES["qf"], QF_PAIRS, r16_w, "SF")
    sf_w  = section("Semifinals", MATCH_NAMES["sf"], SF_PAIRS, qf_w, "Final")

    # Final
    a, b = sf_w[0], sf_w[1]
    champ, runner, p = fav(a, b, "Win", df)
    lines += [
        "— Final —\n",
        f"  Final   {champ} ({p:.0%})  def. {runner}",
        "",
        "=" * 60,
        f"  PREDICTED CHAMPION: {champ}",
        "=" * 60,
    ]

    output = "\n".join(lines)
    print(output)

    out_path = OUTPUT_DIR / "predicted_bracket.txt"
    out_path.write_text(output + "\n")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
