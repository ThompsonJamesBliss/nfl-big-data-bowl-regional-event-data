"""Open but Ignored: run the full pipeline.

    python -m src.run            # uses cache/ if present
    python -m src.run --refresh  # rebuild the tracking cache (about 10 s)

Writes charts and tables to outputs/.
"""
import sys
from pathlib import Path

import pandas as pd

from src import plots
from src.data import load_tables, routes_at_throw
from src.metric import (build_plays, build_routes, flag_ignored, qb_table, receiver_table, sensitivity,
                        situation_summary)

OUT = Path(__file__).resolve().parent.parent / "outputs"


def main(refresh=False):
    OUT.mkdir(exist_ok=True)
    t = load_tables()
    if refresh:
        routes_at_throw(t["pffScoutingData"], refresh=True)

    r = build_routes(t)
    p = build_plays(r)
    r = flag_ignored(r, p)

    s = situation_summary(p)
    rt = receiver_table(r)
    qt = qb_table(p)
    sens = sensitivity(r)

    with pd.option_context("display.width", 160, "display.precision", 3):
        print(f"\n{len(p):,} thrown passes with an identified target\n")
        print(s, "\n")
        print("Miss rate (share of throws with an open man ignored):")
        print("  pressured vs clean:", p.groupby("pressured").miss.mean().round(3).to_dict())
        print("  by coverage:      ", p.groupby("pff_passCoverageType").miss.mean().round(3).to_dict(), "\n")
        print("Sensitivity (first-down rates):")
        print(sens.round(3), "\n")

    s.to_csv(OUT / "situations.csv")
    sens.to_csv(OUT / "sensitivity.csv")
    rt.to_csv(OUT / "receivers.csv", index=False)
    qt.to_csv(OUT / "qbs.csv", index=False)

    plots.cost_chart(s, OUT / "1_cost.png")
    plots.receiver_chart(rt, OUT / "2_receivers.png")
    plots.qb_chart(qt, OUT / "3_qbs.png")
    plots.play_chart(p, r, OUT / "4_play.png", receiver="Justin Jefferson")

    # Sanity checks: fail loudly if the pipeline drifts.
    assert len(p) > 7000, "too few plays with an identified target"
    assert s.loc["Threw to the open man", "first_down"] > s.loc["Open man ignored", "first_down"]
    assert rt.qualified.sum() > 100
    assert (sens.gap > 0).all(), "headline gap flips sign under some definition"


if __name__ == "__main__":
    main(refresh="--refresh" in sys.argv)
