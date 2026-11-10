"""Charts for Open but Ignored."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from src.data import load_play

C_OPEN, C_IGN, C_NONE, C_GREY = "#2a9d8f", "#e76f51", "#8d99ae", "#adb5bd"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def cost_chart(summary, path):
    s = summary
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), sharey=True)
    labels = ["Threw to the\nopen man", "Open man\nignored", "No open option\npast the sticks"]
    colors = [C_OPEN, C_IGN, C_NONE]
    for ax, col, title, fmt in [
        (axes[0], "first_down", "First-down rate", "{:.0%}"),
        (axes[1], "completion", "Completion rate", "{:.0%}"),
        (axes[2], "yards", "Yards per throw", "{:.1f}"),
    ]:
        ax.barh(labels, s[col], color=colors)
        for i, v in enumerate(s[col]):
            ax.text(v, i, " " + fmt.format(v), va="center", fontsize=11)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_xlim(0, s[col].max() * 1.25)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    for i, n in enumerate(s.plays):
        axes[0].text(-0.01, i + 0.33, f"n = {n:,}", ha="right", fontsize=8, color="grey",
                     transform=axes[0].get_yaxis_transform())
    miss_share = s.loc["Open man ignored", "plays"] / s.plays.sum()
    fig.suptitle(f"On {miss_share:.0%} of throws a receiver was open past the sticks and the ball went elsewhere.\n"
                 f"When QBs did throw to the open man, {s.loc['Threw to the open man', 'first_down']:.0%} became first downs; "
                 f"when they looked past him, {s.loc['Open man ignored', 'first_down']:.0%} did.",
                 x=0.02, ha="left", fontsize=13, fontweight="bold", y=1.08)
    fig.text(0.02, -0.04, "Open = 3+ yards from the nearest coverage defender at the throw and 0.5 s before it, "
             "at or past the first-down marker. 2021 Weeks 1-8, thrown passes with an identified target.",
             fontsize=8.5, color="grey")
    _save(fig, path)


def receiver_chart(rt, path, top=15):
    q = rt[rt.qualified].nlargest(top, "ignored").iloc[::-1]
    thrown = q.open_down - q.ignored
    fig, ax = plt.subplots(figsize=(10, 7))
    lab = q.displayName + "  (" + q.officialPosition + ", " + q.possessionTeam + ")"
    ax.barh(lab, q.ignored, color=C_IGN, label="Open and ignored (ball went to a covered or short receiver)")
    ax.barh(lab, thrown, left=q.ignored, color=C_GREY,
            label="Open and targeted, or another open man was targeted")
    for i, (_, r) in enumerate(q.iterrows()):
        ax.text(r.open_down + 0.5, i, f"{r.ignored:.0f} of {r.open_down:.0f}", va="center", fontsize=9)
    ax.set(xlabel="Routes open past the sticks at the throw", ylabel="")
    ax.set_xlim(0, q.open_down.max() * 1.15)
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.08), ncol=1, fontsize=9, frameon=False)
    ax.set_title(f"Most first-down chances left on the field, by receiver (100+ routes)",
                 loc="left", fontweight="bold")
    _save(fig, path)


def qb_chart(qt, path):
    q = qt[qt.qualified].sort_values("take_rate")
    fig, ax = plt.subplots(figsize=(9, 10))
    avg = q.took.sum() / q.options.sum()
    colors = [C_OPEN if v >= avg else C_IGN for v in q.take_rate]
    ax.barh(q.qb + " (" + q.possessionTeam + ")", q.take_rate, color=colors)
    ax.axvline(avg, color="k", ls="--", lw=1)
    ax.text(avg, len(q) - 0.2, f" league {avg:.0%}", fontsize=9, va="bottom")
    for i, (_, r) in enumerate(q.iterrows()):
        ax.text(r.take_rate + 0.005, i, f"{r.took:.0f}/{r.options:.0f}", va="center", fontsize=8, color="grey")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(xlabel="Share of plays with an open man past the sticks where the QB threw to him", ylabel="")
    ax.set_title("Which QBs find the open man past the sticks? (100+ throws)", loc="left", fontweight="bold")
    _save(fig, path)


def play_chart(p, r, path, receiver=None):
    """Draw one miss play at the moment of the throw."""
    m = p[p.miss & ~p.first_down]
    ign = r[r.ignored].merge(m[["gameId", "playId"]], on=["gameId", "playId"])
    if receiver:
        ign = ign[ign.displayName == receiver]
    ign = ign[ign.depth.between(8, 25)].sort_values("sep_throw", ascending=False)
    row = ign.iloc[0]
    play = p[(p.gameId == row.gameId) & (p.playId == row.playId)].iloc[0]
    t = load_play(row.gameId, row.playId)
    snap, end = int(row.snap), int(row.end)
    at = t[t.frameId == end]
    off_team = play.possessionTeam
    los, sticks = row.los, row.los + play.yardsToGo

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.add_patch(patches.Rectangle((0, 0), 120, 53.3, color="#f4f1ea", zorder=0))
    for x in range(10, 111, 5):
        ax.axvline(x, color="white", lw=1.2, zorder=1)
    ax.axvline(los, color="#457b9d", lw=2, zorder=2)
    ax.axvline(sticks, color="#f4a261", lw=2, zorder=2)
    ax.text(los, 52.5, " line of scrimmage", color="#457b9d", fontsize=9, va="top")
    ax.text(sticks, 50.5, " first-down marker", color="#e76f51", fontsize=9, va="top")

    def path_of(nid, color):
        d = t[(t.nflId == nid) & t.frameId.between(snap, end)]
        ax.plot(d.x, d.y, color=color, lw=2, zorder=3)

    tgt_id = r[(r.gameId == row.gameId) & (r.playId == row.playId) & (r.targeted == 1)].nflId.iloc[0]
    path_of(row.nflId, C_IGN)
    path_of(tgt_id, "#264653")

    pl = at[at.team != "football"]
    off = pl[pl.team == off_team]
    dfn = pl[pl.team != off_team]
    ax.scatter(dfn.x, dfn.y, s=110, color="white", edgecolor="#6c757d", lw=1.5, zorder=4, label="Defense")
    ax.scatter(off.x, off.y, s=110, color="#6c757d", zorder=4, label="Offense")
    ball = at[at.team == "football"]
    ax.scatter(ball.x, ball.y, s=40, color="#8b4513", zorder=6, label="Ball")

    rec = at[at.nflId == row.nflId].iloc[0]
    tg = at[at.nflId == tgt_id].iloc[0]
    ax.scatter(rec.x, rec.y, s=320, color=C_IGN, edgecolor="k", zorder=5)
    ax.scatter(tg.x, tg.y, s=320, color="#264653", edgecolor="k", zorder=5)
    tgt_row = r[(r.gameId == row.gameId) & (r.playId == row.playId) & (r.nflId == tgt_id)].iloc[0]
    ax.annotate(f"{row.displayName}: OPEN, NOT TARGETED\n{row.sep_throw:.1f} yds of space, {row.depth:.0f} yds downfield",
                (rec.x, rec.y), xytext=(10, 18), textcoords="offset points", fontsize=10, color=C_IGN,
                fontweight="bold", zorder=7)
    ax.annotate(f"Target: {tgt_row.displayName}\n{tgt_row.sep_throw:.1f} yds of space, {tgt_row.depth:.0f} yds downfield",
                (tg.x, tg.y), xytext=(10, -34), textcoords="offset points", fontsize=10, color="#264653",
                fontweight="bold", zorder=7)

    xs = pl.x
    ax.set(xlim=(max(0, xs.min() - 6), min(120, xs.max() + 18)), ylim=(0, 53.3), aspect="equal",
           xticks=[], yticks=[])
    ax.legend(loc="lower left", fontsize=8, frameon=True)
    desc = play.playDescription if len(play.playDescription) < 120 else play.playDescription[:117] + "..."
    qtr = "OT" if play.quarter == 5 else f"Q{play.quarter}"
    ax.set_title(f"{off_team} vs {play.defensiveTeam}, {qtr} {play.gameClock}, "
                 f"{int(play.down)} & {int(play.yardsToGo)}  |  result: {play.prePenaltyPlayResult:+.0f} yds, no first down\n"
                 f"{desc}", loc="left", fontsize=10)
    _save(fig, path)
    return row.gameId, row.playId
