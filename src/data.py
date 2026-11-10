"""Load the tables, identify each pass's intended receiver, and measure every route at the throw."""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = ROOT / "cache" / "routes_at_throw.pkl"

SNAP = {"ball_snap", "autoevent_ballsnap"}
END = {"pass_forward", "autoevent_passforward", "qb_sack", "qb_strip_sack", "run"}
PRE_FRAMES = 5  # 0.5 s at 10 Hz: "was he still open while the QB was deciding?"


def load_tables():
    return {name: pd.read_csv(DATA / f"{name}.csv")
            for name in ("games", "plays", "players", "pffScoutingData")}


def _name_key(first, last):
    """First initial + last name, lower case, letters only: 'A.St. Brown' and 'Amon-Ra St. Brown' -> 'A|stbrown'."""
    return first.str[0] + "|" + last.str.replace(r"[^A-Za-z]", "", regex=True).str.lower()


def match_targets(plays, players, scout):
    """Return (gameId, playId, nflId) of the intended receiver, parsed from playDescription.

    Descriptions read 'pass short right to C.Godwin' or 'pass incomplete deep left intended for A.Cooper'.
    The name is matched to a route runner on the same play. Plays where the description names no
    receiver ('pass incomplete short right.') are dropped.
    """
    pat = (r"pass (?:incomplete )?(?:short|deep)? ?(?:left|middle|right)? ?(?:to|intended for) "
           r"([A-Z][a-z]?\.(?:St\. )?[A-Z][A-Za-z'\-]+)")
    tok = plays.playDescription.str.extract(pat)[0].str.split(".", n=1)
    p = plays[["gameId", "playId"]].assign(key=_name_key(tok.str[0], tok.str[1])).dropna()

    nm = players.displayName.str.replace(r" (Jr|Sr|II|III|IV)\.?$", "", regex=True)
    nm = nm.str.replace(r"^((?:[A-Z]\.)+) ?", lambda m: m.group(1).replace(".", "") + " ", regex=True)
    parts = nm.str.split(n=1)
    pk = players[["nflId"]].assign(key=_name_key(parts.str[0], parts.str[1]))

    rr = scout.loc[scout.pff_role == "Pass Route", ["gameId", "playId", "nflId"]].merge(pk, on="nflId")
    t = p.merge(rr, on=["gameId", "playId", "key"])
    t = t[~t.duplicated(["gameId", "playId"], keep=False)]  # ambiguous name on the play: drop
    return t[["gameId", "playId", "nflId"]]


def _one_game(path, roles):
    t = pd.read_csv(path, usecols=["gameId", "playId", "nflId", "frameId", "team",
                                   "playDirection", "x", "y", "s", "dir", "event"])
    t["event"] = t["event"].fillna("None")  # some files use NaN for untagged frames, others "None"
    left = t.playDirection == "left"  # normalize so the offense always moves toward +x
    t.loc[left, "x"] = 120 - t.loc[left, "x"]
    t.loc[left, "y"] = 53.3 - t.loc[left, "y"]
    t.loc[left, "dir"] = (t.loc[left, "dir"] + 180) % 360
    rad = np.deg2rad(t["dir"])  # dir: 0 deg = +y, 90 deg = +x
    t["vx"], t["vy"] = t.s * np.sin(rad), t.s * np.cos(rad)

    ev = t[["playId", "frameId", "event"]].drop_duplicates()
    snap = ev[ev.event.isin(SNAP)].groupby("playId").frameId.min().rename("snap")
    ev = ev.join(snap, on="playId")
    end_rows = ev[ev.event.isin(END) & (ev.frameId > ev.snap)].sort_values("frameId").drop_duplicates("playId")
    win = end_rows.set_index("playId")[["snap", "frameId", "event"]].rename(
        columns={"frameId": "end", "event": "end_event"}).dropna(subset=["snap"])
    win["pre"] = np.maximum(win.snap, win.end - PRE_FRAMES)

    ball = t[t.team == "football"].merge(win.reset_index(), on="playId")
    los = ball[ball.frameId == ball.snap].set_index("playId").x.rename("los")

    frames = pd.concat([
        win.reset_index()[["playId", "end"]].rename(columns={"end": "frameId"}).assign(when="throw"),
        win.reset_index()[["playId", "pre"]].rename(columns={"pre": "frameId"}).assign(when="pre"),
    ])
    x = t.merge(frames, on=["playId", "frameId"]).merge(roles, on=["gameId", "playId", "nflId"])
    rec = x[x.pff_role == "Pass Route"]
    dfn = x.loc[x.pff_role == "Coverage", ["playId", "when", "nflId", "x", "y", "vx", "vy"]]
    j = rec.merge(dfn, on=["playId", "when"], suffixes=("", "_d"))
    j["sep"] = np.hypot(j.x - j.x_d, j.y - j.y_d)
    j = j.loc[j.groupby(["playId", "when", "nflId"]).sep.idxmin()]
    # Closing speed (yd/s): how fast the gap to the nearest defender is shrinking. Positive = closing.
    ux, uy = (j.x - j.x_d) / j.sep, (j.y - j.y_d) / j.sep
    j["closing"] = (j.vx_d - j.vx) * ux + (j.vy_d - j.vy) * uy

    w = j.set_index(["gameId", "playId", "nflId", "when"])[["sep", "x", "y", "s", "closing", "nflId_d"]].unstack("when")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index().rename(columns={"nflId_d_throw": "defId"}).drop(
        columns=["x_pre", "y_pre", "s_pre", "closing_pre", "nflId_d_pre"])
    w = w.join(win[["snap", "end", "end_event"]], on="playId").join(los, on="playId")
    w["depth"] = w.x_throw - w.los
    w["ttt"] = (w.end - w.snap) / 10
    return w


def routes_at_throw(scout, refresh=False):
    """One row per route runner per play: nearest-defender separation at the throw and 0.5 s before,
    position at the throw, depth past the line of scrimmage, time to throw. Cached."""
    if CACHE.exists() and not refresh:
        return pd.read_pickle(CACHE)
    roles = scout.loc[scout.pff_role.isin(["Pass Route", "Coverage"]), ["gameId", "playId", "nflId", "pff_role"]]
    out = pd.concat([_one_game(f, roles) for f in sorted(glob.glob(str(DATA / "tracking" / "*.csv")))],
                    ignore_index=True)
    os.makedirs(CACHE.parent, exist_ok=True)
    out.to_pickle(CACHE)
    return out


def load_play(gameId, playId):
    """All tracking rows for one play, direction-normalized."""
    t = pd.read_csv(DATA / "tracking" / f"tracking_{gameId}.csv").query("playId == @playId").copy()
    t["event"] = t["event"].fillna("None")
    left = t.playDirection == "left"
    t.loc[left, "x"] = 120 - t.loc[left, "x"]
    t.loc[left, "y"] = 53.3 - t.loc[left, "y"]
    return t
