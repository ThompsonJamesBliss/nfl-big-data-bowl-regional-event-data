"""Openness score: how open a receiver is at the throw, on a 0-100 scale.

The score is the model's probability that a throw to this receiver, at this moment, is completed.
It is trained on targeted routes, then applied to every route (targeted or not), so it measures how
throwable each receiver was regardless of where the ball went.

Inputs, all measured at the throw:
  sep_throw      yards to the nearest coverage defender
  sep_pre        same, 0.5 s before the throw
  closing_throw  how fast that defender is closing the gap (yd/s, positive = closing)
  depth          yards past the line of scrimmage
  position (WR/TE/RB) and coverage type (man/zone/other)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SKILL = ["WR", "TE", "RB"]


def features(r):
    d = pd.DataFrame(index=r.index)
    d["log_sep"] = np.log1p(r.sep_throw)
    d["log_sep_pre"] = np.log1p(r.sep_pre)
    d["closing"] = r.closing_throw.clip(-8, 8)
    d["depth"] = r.depth.clip(-10, 40)
    d["depth_sq"] = d.depth ** 2 / 100
    d["behind_los"] = (r.depth < 0).astype(float)
    d["sep_x_depth"] = d.log_sep * d.depth / 10
    for p in ("TE", "RB"):
        d[f"pos_{p}"] = (r.officialPosition == p).astype(float)
    for c in ("Man", "Other"):
        d[f"cov_{c}"] = (r.pff_passCoverageType == c).astype(float)
    return d


BASELINE = ["log_sep"]


def _model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))


def evaluate(r):
    """5-fold cross-validation grouped by game, comparing the full model with separation alone."""
    t = r[(r.targeted == 1) & r.passResult.isin(["C", "I", "IN"]) & r.officialPosition.isin(SKILL)]
    X, y, g = features(t), (t.passResult == "C").astype(int), t.gameId
    out = {}
    for name, cols in (("separation only", BASELINE), ("openness model", list(X.columns))):
        pred = np.zeros(len(t))
        for tr, te in GroupKFold(5).split(X, y, g):
            m = _model().fit(X.iloc[tr][cols], y.iloc[tr])
            pred[te] = m.predict_proba(X.iloc[te][cols])[:, 1]
        out[name] = {"AUC": roc_auc_score(y, pred), "log loss": log_loss(y, pred)}
    out["always predict the average"] = {"AUC": 0.5, "log loss": log_loss(y, np.full(len(y), y.mean()))}
    return pd.DataFrame(out).T, t.assign(cv_pred=pred)


def fit_and_score(r):
    """Fit on all targeted routes, add `openness` (0-100) to every skill-position route."""
    t = r[(r.targeted == 1) & r.passResult.isin(["C", "I", "IN"]) & r.officialPosition.isin(SKILL)]
    m = _model().fit(features(t), (t.passResult == "C").astype(int))
    r = r[r.officialPosition.isin(SKILL)].copy()
    r["openness"] = 100 * m.predict_proba(features(r))[:, 1]
    coefs = pd.Series(m[-1].coef_[0], index=features(t).columns).sort_values()
    return r, coefs


def receiver_scores(r, min_routes=100):
    """Per receiver: average openness on all routes and on routes at or past the first-down marker."""
    g = r.groupby(["nflId", "displayName", "officialPosition", "possessionTeam"])
    out = g.agg(routes=("openness", "size"), targets=("targeted", "sum"),
                openness=("openness", "mean")).reset_index()
    down = r[r.past_sticks].groupby("nflId").agg(routes_past_sticks=("openness", "size"),
                                                 openness_past_sticks=("openness", "mean"))
    out = out.join(down, on="nflId")
    pos_avg = out[out.routes >= min_routes].groupby("officialPosition").openness.mean()
    out["openness_vs_position"] = out.openness - out.officialPosition.map(pos_avg)
    out["qualified"] = out.routes >= min_routes
    return out.sort_values("openness_past_sticks", ascending=False)


def best_option_gap(r):
    """Per play: openness of the most open receiver past the sticks vs the receiver actually targeted."""
    tgt = r[r.targeted == 1].set_index(["gameId", "playId"]).openness.rename("target_openness")
    best = r[r.past_sticks & (r.targeted == 0)].groupby(["gameId", "playId"]).openness.max().rename("best_other_past_sticks")
    p = pd.concat([tgt, best], axis=1).dropna()
    p["gap"] = p.best_other_past_sticks - p.target_openness
    return p
