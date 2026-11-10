"""Per-route summary at the moment of the throw, built once from all tracking files.

One row per route runner per play: separation to the nearest coverage defender at the
end of the pass-play window, who that defender is, route depth, speed, and time to throw.
Used by eda.ipynb. Cached to cache/eda_routes.pkl.
"""
import glob
import os

import numpy as np
import pandas as pd

SNAP = {"ball_snap", "autoevent_ballsnap"}
END = {"pass_forward", "autoevent_passforward", "qb_sack", "qb_strip_sack", "run"}
CACHE = "cache/eda_routes.pkl"


def _one_game(path, roles):
    t = pd.read_csv(path, usecols=["gameId", "playId", "nflId", "frameId", "team",
                                   "playDirection", "x", "y", "s", "event"])
    t["event"] = t["event"].fillna("None")  # some files store untagged frames as NaN, others as "None"
    left = t["playDirection"] == "left"
    t.loc[left, "x"] = 120 - t.loc[left, "x"]
    t.loc[left, "y"] = 53.3 - t.loc[left, "y"]

    ev = t[["playId", "frameId", "event"]].drop_duplicates()
    snap = ev[ev.event.isin(SNAP)].groupby("playId").frameId.min().rename("snap")
    ev = ev.merge(snap, on="playId")
    end = ev[ev.event.isin(END) & (ev.frameId > ev.snap)].groupby("playId").frameId.min().rename("end")
    win = pd.concat([snap, end], axis=1).dropna().astype(int).reset_index()
    win["end_event"] = win.merge(ev, left_on=["playId", "end"], right_on=["playId", "frameId"]) \
                          .groupby("playId").event.first().reindex(win.playId).values

    los = t[t.team == "football"].merge(win, on="playId").query("frameId == snap")[["playId", "x"]] \
        .rename(columns={"x": "los"})
    at_end = t.merge(win, on="playId").query("frameId == end")
    at_end = at_end.merge(roles, on=["gameId", "playId", "nflId"])

    rec = at_end[at_end.pff_role == "Pass Route"]
    dfn = at_end[at_end.pff_role == "Coverage"][["playId", "nflId", "x", "y"]]
    j = rec.merge(dfn, on="playId", suffixes=("", "_d"))
    j["dist"] = np.hypot(j.x - j.x_d, j.y - j.y_d)
    j = j.loc[j.groupby(["playId", "nflId"]).dist.idxmin()]
    j = j.merge(los, on="playId")
    j["depth"] = j.x - j.los
    j["ttt"] = (j.end - j.snap) / 10
    return j[["gameId", "playId", "nflId", "nflId_d", "dist", "depth", "s", "ttt", "end_event"]] \
        .rename(columns={"nflId_d": "defId", "dist": "sep"})


def build_routes(scout):
    if os.path.exists(CACHE):
        return pd.read_pickle(CACHE)
    roles = scout.loc[scout.pff_role.isin(["Pass Route", "Coverage"]), ["gameId", "playId", "nflId", "pff_role"]]
    out = pd.concat([_one_game(f, roles) for f in sorted(glob.glob("data/tracking/*.csv"))], ignore_index=True)
    os.makedirs("cache", exist_ok=True)
    out.to_pickle(CACHE)
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    r = build_routes(pd.read_csv("data/pffScoutingData.csv"))
    print(r.shape, r.groupby(["gameId", "playId"]).ngroups, "plays", f"{time.time() - t0:.0f}s")
    print(r.describe().round(2))
    print(r.end_event.value_counts())
