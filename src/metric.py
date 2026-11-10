"""Open but Ignored: receivers who were open past the sticks when the ball went somewhere else.

A route is **open downfield** at the throw when the receiver
  - is at least OPEN_YDS from the nearest coverage defender at the throw,
  - was also at least OPEN_YDS clear 0.5 s earlier (open while the QB was deciding, not just
    drifting free as the ball left), and
  - is at or past the first-down marker.

A play is a **miss** when at least one non-targeted receiver was open downfield and the
targeted receiver was not. Those non-targeted receivers are **open but ignored**.
"""
import pandas as pd

OPEN_YDS = 3.0
THROWS = {"pass_forward", "autoevent_passforward"}
SKILL = ["WR", "TE", "RB"]


def build_routes(t):
    """Route table for thrown passes with one identified target."""
    plays, players, scout = t["plays"], t["players"], t["pffScoutingData"]
    from src.data import match_targets, routes_at_throw

    r = routes_at_throw(scout)
    r = r[r.end_event.isin(THROWS)]
    tg = match_targets(plays, players, scout).assign(targeted=1)
    r = r.merge(tg, on=["gameId", "playId", "nflId"], how="left").fillna({"targeted": 0})
    r = r[r.groupby(["gameId", "playId"]).targeted.transform("sum") == 1]  # target must be identified

    p = plays[["gameId", "playId", "possessionTeam", "defensiveTeam", "down", "yardsToGo", "passResult",
               "prePenaltyPlayResult", "pff_passCoverageType", "playDescription", "quarter", "gameClock"]]
    r = r.merge(p, on=["gameId", "playId"])
    r = r.merge(players[["nflId", "displayName", "officialPosition"]], on="nflId")

    passer = scout.loc[scout.pff_role == "Pass", ["gameId", "playId", "nflId"]].drop_duplicates(["gameId", "playId"])
    passer = passer.merge(players[["nflId", "displayName"]], on="nflId").rename(
        columns={"nflId": "qbId", "displayName": "qb"})
    r = r.merge(passer, on=["gameId", "playId"], how="left")

    rush = scout[scout.pff_role == "Pass Rush"].groupby(["gameId", "playId"])[
        ["pff_hit", "pff_hurry", "pff_sack"]].max().fillna(0).max(axis=1).rename("pressured")
    r = r.join(rush.astype(bool), on=["gameId", "playId"]).fillna({"pressured": False})

    r["past_sticks"] = r.depth >= r.yardsToGo
    return set_open(r, OPEN_YDS)


def set_open(r, open_yds, require_pre=True):
    r = r.copy()
    clear = r.sep_throw >= open_yds
    if require_pre:
        clear &= r.sep_pre >= open_yds
    r["open_down"] = clear & r.past_sticks
    return r


def sensitivity(r):
    """Re-run the headline comparison under other definitions of 'open' and in man coverage only."""
    rows = []
    variants = [(f"{y:g}+ yds, also 0.5 s before", y, True, None) for y in (2, 3, 4, 5)]
    variants += [("3+ yds at the throw only", 3, False, None),
                 ("3+ yds, man coverage only", 3, True, "Man"),
                 ("3+ yds, zone coverage only", 3, True, "Zone")]
    for label, yds, pre, cov in variants:
        rr = r if cov is None else r[r.pff_passCoverageType == cov]
        p = build_plays(set_open(rr, yds, pre))
        s = p.groupby("situation").first_down.mean()
        rows.append({"definition": label, "throws": len(p), "miss_share": p.miss.mean(),
                     "fd_threw_open": s.get("Threw to the open man"),
                     "fd_ignored": s.get("Open man ignored"),
                     "fd_no_option": s.get("No open option past the sticks")})
    out = pd.DataFrame(rows).set_index("definition")
    out["gap"] = out.fd_threw_open - out.fd_ignored
    return out


def build_plays(r):
    """One row per play: what the QB did, and whether an open-downfield option went unused."""
    tgt = r[r.targeted == 1].set_index(["gameId", "playId"])
    other_open = r[(r.targeted == 0) & r.open_down].groupby(["gameId", "playId"]).size().rename("n_ignored")
    p = tgt.join(other_open).fillna({"n_ignored": 0})
    p["target_open_down"] = p.open_down
    p["miss"] = (p.n_ignored > 0) & ~p.target_open_down
    p["situation"] = "No open option past the sticks"
    p.loc[p.target_open_down, "situation"] = "Threw to the open man"
    p.loc[p.miss, "situation"] = "Open man ignored"
    p["first_down"] = p.prePenaltyPlayResult >= p.yardsToGo
    return p.reset_index()


def flag_ignored(r, p):
    """Mark each route that was open downfield on a miss play."""
    miss = p.loc[p.miss, ["gameId", "playId"]].assign(miss=True)
    r = r.merge(miss, on=["gameId", "playId"], how="left").fillna({"miss": False})
    r["ignored"] = r.miss & r.open_down & (r.targeted == 0)
    return r


def situation_summary(p):
    order = ["Threw to the open man", "Open man ignored", "No open option past the sticks"]
    return p.groupby("situation").agg(
        plays=("first_down", "size"),
        completion=("passResult", lambda s: (s == "C").mean()),
        interception=("passResult", lambda s: (s == "IN").mean()),
        first_down=("first_down", "mean"),
        yards=("prePenaltyPlayResult", "mean"),
        target_depth=("depth", "median"),
        pressured=("pressured", "mean"),
        time_to_throw=("ttt", "median"),
    ).loc[order]


def receiver_table(r, min_routes=100):
    g = r[r.officialPosition.isin(SKILL)].groupby(["nflId", "displayName", "officialPosition", "possessionTeam"])
    out = g.agg(routes=("open_down", "size"), targets=("targeted", "sum"),
                open_down=("open_down", "sum"), ignored=("ignored", "sum")).reset_index()
    out["target_rate"] = out.targets / out.routes
    out["open_down_rate"] = out.open_down / out.routes
    out["ignored_share"] = out.ignored / out.open_down.where(out.open_down > 0)
    out["qualified"] = out.routes >= min_routes
    return out.sort_values("ignored", ascending=False)


def qb_table(p, min_plays=100):
    has_option = p.target_open_down | p.miss
    g = p.assign(has_option=has_option).groupby(["qbId", "qb", "possessionTeam"])
    out = g.agg(throws=("miss", "size"), options=("has_option", "sum"),
                took=("target_open_down", "sum"), missed=("miss", "sum"),
                pressured=("pressured", "mean"), ttt=("ttt", "median")).reset_index()
    out["take_rate"] = out.took / out.options.where(out.options > 0)
    out["miss_rate"] = out.missed / out.throws
    out["qualified"] = out.throws >= min_plays
    return out.sort_values("miss_rate", ascending=False)
