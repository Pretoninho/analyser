"""
Analyse approfondie de la macro 09:50 (mac_idx=2) — full dataset + charts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import pytz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from data.binance import load_binance_1m
from engine.stats_state import (
    MACROS, compute_daily_context, compute_pool_ctx, build_weekly_levels,
)
from main import _sim_trade_rr

MAC_START  = 590
EXIT_HM    = 960
SL_PCT     = 0.006
RR         = 2.5
FEE_RATE   = 0.0005
SLIPPAGE   = 0.0002
REF_WINDOW = 240
SKIP_DAYS  = {0}

OUT_DIR = Path("display/analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Chargement ───────────────────────────────────────────────────
df = load_binance_1m()
df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
et_tz = pytz.timezone("America/New_York")
df["ts_et"]   = df["timestamp"].dt.tz_convert(et_tz)
df["hm_et"]   = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
df["date_et"] = df["ts_et"].dt.date
df = df[df["hm_et"] < 20 * 60]

weekly = build_weekly_levels(df)

episodes, dates = [], []
for date, grp in df.groupby("date_et"):
    hm_vals = set(grp["hm_et"].values)
    if (any(60  <= h < 300 for h in hm_vals) and
        any(420 <= h < 600 for h in hm_vals) and
        any(530 <= h < 910 for h in hm_vals) and
        len(grp) >= 60):
        episodes.append(grp.sort_values("ts_et").reset_index(drop=True))
        dates.append(date)

print(f"[analyse] {len(episodes)} jours totaux")

# ── Collecte des setups 09:50 ────────────────────────────────────
records = []
for day_df, date in zip(episodes, dates):
    if date.weekday() in SKIP_DAYS:
        continue

    pwh, pwl = weekly.get(date, (None, None))
    ctx = compute_daily_context(day_df, pwh=pwh, pwl=pwl)
    mc_ = ctx["month_ctx"]
    dc_ = ctx["day_ctx"]
    lc_ = ctx["london_ctx"]

    pre_start = MAC_START - 20
    pre_mask  = (day_df["hm_et"] >= pre_start) & (day_df["hm_et"] < MAC_START)
    exit_mask = (day_df["hm_et"] >= MAC_START)  & (day_df["hm_et"] < EXIT_HM)

    pre_df  = day_df[pre_mask]
    exit_df = day_df[exit_mask]
    if len(pre_df) < 3 or len(exit_df) < 5:
        continue

    pre_high = float(pre_df["high"].max())
    pre_low  = float(pre_df["low"].min())
    first    = exit_df.iloc[0]

    if float(first["high"]) > pre_high:
        sc = 1
    elif float(first["low"]) < pre_low:
        sc = 2
    else:
        continue  # aligned_only

    ref_mask = (day_df["hm_et"] >= max(0, pre_start - REF_WINDOW)) & (day_df["hm_et"] < pre_start)
    ref_df   = day_df[ref_mask]
    if len(ref_df) >= 5:
        ref_h = float(ref_df["high"].max())
        ref_l = float(ref_df["low"].min())
    else:
        ref_h = ctx.get("session_high")
        ref_l = ctx.get("session_low")

    pc = compute_pool_ctx(pre_high, pre_low, ref_h, ref_l, pwh, pwl)
    entry_px   = float(first["open"])
    sweep_high = max(float(first["high"]), pre_high)
    sweep_low  = min(float(first["low"]),  pre_low)

    if sc == 2:
        direction = +1
        if pc == 2 and ref_h is not None and ref_h > entry_px:
            tp_pct  = (ref_h - entry_px) / entry_px
            sl_pct_ = SL_PCT + max(0.0, (entry_px - sweep_low) / entry_px)
        else:
            tp_pct  = SL_PCT * RR
            sl_pct_ = SL_PCT
        if tp_pct <= sl_pct_ or sl_pct_ <= 0:
            continue
        pnl, reason, _, _, _, nc = _sim_trade_rr(
            exit_df, entry_px * (1 + SLIPPAGE), +1, sl_pct_, tp_pct, FEE_RATE, SLIPPAGE, verbose=True)
    else:
        direction = -1
        if pc == 1 and ref_l is not None and ref_l < entry_px:
            tp_pct  = (entry_px - ref_l) / entry_px
            sl_pct_ = SL_PCT + max(0.0, (sweep_high - entry_px) / entry_px)
        else:
            tp_pct  = SL_PCT * RR
            sl_pct_ = SL_PCT
        if tp_pct <= sl_pct_ or sl_pct_ <= 0:
            continue
        pnl, reason, _, _, _, nc = _sim_trade_rr(
            exit_df, entry_px * (1 - SLIPPAGE), -1, sl_pct_, tp_pct, FEE_RATE, SLIPPAGE, verbose=True)

    records.append({
        "date":   date,
        "year":   date.year,
        "month":  date.month,
        "dow":    date.weekday(),
        "mc":     mc_,
        "dc":     dc_,
        "lc":     lc_,
        "sc":     sc,
        "pc":     pc,
        "dir":    "LONG" if direction == 1 else "SHORT",
        "pnl":    pnl,
        "reason": reason,
        "nc":     nc,
    })

df_t = pd.DataFrame(records)
N    = len(df_t)
print(f"[analyse] {N} setups 09:50 (full dataset, sans lundi, aligned_only)")

# ── Helpers ──────────────────────────────────────────────────────
def stats(sub):
    if len(sub) == 0:
        return dict(n=0, wr=0.0, avg=0.0, pf=0.0)
    arr  = sub["pnl"].values
    wins = arr[arr > 0]; losses = arr[arr < 0]
    pf   = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")
    return dict(n=len(sub), wr=round(len(wins)/len(sub)*100, 1),
                avg=round(arr.mean()*100, 3), pf=round(pf, 3))

lc_names  = {0: "NO_RAID", 1: "RAID_H", 2: "RAID_L"}
sc_names  = {1: "SW_H->SHORT", 2: "SW_L->LONG"}
pc_names  = {0: "NEUTRAL", 1: "BSL_swept", 2: "SSL_swept"}
mc_names  = {0: "Mois WEAK", 1: "Mois NEUTRAL", 2: "Mois STRONG"}
day_names = {1: "Mardi", 2: "Mercredi", 3: "Jeudi", 4: "Vendredi", 5: "Samedi", 6: "Dimanche"}

# ── Synthese console ─────────────────────────────────────────────
print("\n" + "="*62)
print("  MACRO 09:50 — ANALYSE COMPLETE (train + test, sans lundi)")
print("="*62)

s = stats(df_t)
print(f"\n  GLOBAL : {s['n']} trades | WR {s['wr']}% | Avg {s['avg']:+.3f}% | PF {s['pf']}")

print("\n--- Direction (sweep) ---")
for sc_v, lbl in sc_names.items():
    s = stats(df_t[df_t["sc"] == sc_v])
    print(f"  {lbl:<16} N={s['n']:>4}  WR={s['wr']:>5.1f}%  Avg={s['avg']:>+6.3f}%  PF={s['pf']:.3f}")

print("\n--- London context ---")
for lc_v, lbl in lc_names.items():
    s = stats(df_t[df_t["lc"] == lc_v])
    print(f"  {lbl:<12} N={s['n']:>4}  WR={s['wr']:>5.1f}%  Avg={s['avg']:>+6.3f}%  PF={s['pf']:.3f}")

print("\n--- Pool context ---")
for pc_v, lbl in pc_names.items():
    s = stats(df_t[df_t["pc"] == pc_v])
    print(f"  {lbl:<14} N={s['n']:>4}  WR={s['wr']:>5.1f}%  Avg={s['avg']:>+6.3f}%  PF={s['pf']:.3f}")

print("\n--- London x Pool (N>=5) ---")
combos = []
for lc_v in range(3):
    for pc_v in range(3):
        s = stats(df_t[(df_t["lc"] == lc_v) & (df_t["pc"] == pc_v)])
        if s["n"] >= 5:
            combos.append((s["avg"], lc_names[lc_v], pc_names[pc_v], s))
combos.sort(reverse=True)
for avg, l, p, s in combos:
    print(f"  {l:<12} x {p:<14}  N={s['n']:>4}  WR={s['wr']:>5.1f}%  Avg={s['avg']:>+6.3f}%  PF={s['pf']:.3f}")

print("\n--- Jour de semaine ---")
for dow_v in sorted(day_names):
    s = stats(df_t[df_t["dow"] == dow_v])
    if s["n"] > 0:
        print(f"  {day_names[dow_v]:<10} N={s['n']:>4}  WR={s['wr']:>5.1f}%  Avg={s['avg']:>+6.3f}%  PF={s['pf']:.3f}")

print("\n--- Mois context ---")
for mc_v, lbl in mc_names.items():
    s = stats(df_t[df_t["mc"] == mc_v])
    print(f"  {lbl:<16} N={s['n']:>4}  WR={s['wr']:>5.1f}%  Avg={s['avg']:>+6.3f}%  PF={s['pf']:.3f}")

print("\n--- Exit reason ---")
for r in ["TP", "SL", "EOD"]:
    sub = df_t[df_t["reason"] == r]
    if len(sub) > 0:
        print(f"  {r:<5} N={len(sub):>4}  Avg={sub['pnl'].mean()*100:>+6.3f}%  Duree moy={sub['nc'].mean():.0f} min")

print("="*62)

# ── CHART 1 : Synthese par contexte ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Macro 09:50 — Analyse par contexte (full dataset, sans lundi)", fontsize=13, fontweight="bold")
COLORS = {"ok": "#4CAF50", "warn": "#FF9800", "bad": "#F44336", "neutral": "#2196F3"}

def color_wr(wr):
    return COLORS["ok"] if wr >= 55 else (COLORS["warn"] if wr >= 45 else COLORS["bad"])

def bar_chart(ax, labels, wrs, avgs, ns, title):
    x = np.arange(len(labels))
    w = 0.35
    colors_wr  = [color_wr(wr) for wr in wrs]
    colors_avg = [COLORS["ok"] if a > 0 else COLORS["bad"] for a in avgs]
    b1 = ax.bar(x - w/2, wrs,  w, color=colors_wr,  alpha=0.85, label="WR%")
    b2 = ax.bar(x + w/2, avgs, w, color=colors_avg, alpha=0.85, label="Avg%")
    ax.axhline(0, color="white", linewidth=0.5, alpha=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(N={n})" for l, n in zip(labels, ns)], fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_facecolor("#1a1a2e")
    for i, (wr, avg) in enumerate(zip(wrs, avgs)):
        ax.text(i - w/2, wr + 0.5, f"{wr:.0f}%", ha="center", fontsize=7, color="white")
        ax.text(i + w/2, avg + (0.1 if avg >= 0 else -0.3), f"{avg:+.2f}%", ha="center", fontsize=7, color="white")

fig.patch.set_facecolor("#0d0d1a")
for ax in axes.flat:
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white", labelsize=8)
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# --- ax[0,0] Direction ---
data_dir = [(sc_names[sc_v], stats(df_t[df_t["sc"] == sc_v])) for sc_v in [1, 2]]
bar_chart(axes[0,0],
    [d[0].replace("->", "\n") for d in data_dir],
    [d[1]["wr"]  for d in data_dir],
    [d[1]["avg"] for d in data_dir],
    [d[1]["n"]   for d in data_dir],
    "Direction (Sweep)")

# --- ax[0,1] London ---
data_lc = [(lc_names[v], stats(df_t[df_t["lc"] == v])) for v in range(3)]
bar_chart(axes[0,1],
    [d[0] for d in data_lc],
    [d[1]["wr"]  for d in data_lc],
    [d[1]["avg"] for d in data_lc],
    [d[1]["n"]   for d in data_lc],
    "London Context")

# --- ax[0,2] Pool ---
data_pc = [(pc_names[v], stats(df_t[df_t["pc"] == v])) for v in range(3)]
bar_chart(axes[0,2],
    [d[0] for d in data_pc],
    [d[1]["wr"]  for d in data_pc],
    [d[1]["avg"] for d in data_pc],
    [d[1]["n"]   for d in data_pc],
    "Pool Context (Liquidite)")


# --- ax[1,0] Jour ---
dow_items = [(day_names[v], stats(df_t[df_t["dow"] == v])) for v in sorted(day_names) if stats(df_t[df_t["dow"] == v])["n"] > 0]
bar_chart(axes[1,0],
    [d[0] for d in dow_items],
    [d[1]["wr"]  for d in dow_items],
    [d[1]["avg"] for d in dow_items],
    [d[1]["n"]   for d in dow_items],
    "Jour de semaine")

# --- ax[1,1] Mois context ---
data_mc = [(mc_names[v], stats(df_t[df_t["mc"] == v])) for v in range(3)]
bar_chart(axes[1,1],
    [d[0].replace(" ", "\n") for d in data_mc],
    [d[1]["wr"]  for d in data_mc],
    [d[1]["avg"] for d in data_mc],
    [d[1]["n"]   for d in data_mc],
    "Mois Context")

# --- ax[1,2] Exit reason ---
data_ex = [(r, stats(df_t[df_t["reason"] == r])) for r in ["TP","SL","EOD"]]
bar_chart(axes[1,2],
    [d[0] for d in data_ex],
    [d[1]["wr"]  for d in data_ex],
    [d[1]["avg"] for d in data_ex],
    [d[1]["n"]   for d in data_ex],
    "Type de sortie")

plt.tight_layout()
p1 = OUT_DIR / "0950_contextes.png"
plt.savefig(p1, dpi=130, facecolor=fig.get_facecolor())
plt.close()
print(f"[chart] {p1}")

# ── CHART 2 : Heatmap London x Pool ─────────────────────────────
fig2, ax2 = plt.subplots(1, 2, figsize=(13, 5))
fig2.patch.set_facecolor("#0d0d1a")
fig2.suptitle("Macro 09:50 — Heatmap London x Pool", fontsize=12, fontweight="bold", color="white")

for i, (metric, title, fmt) in enumerate([
    ("wr", "Win Rate (%)", ".0f"),
    ("avg", "Avg P&L (%)", "+.3f"),
]):
    matrix = np.zeros((3, 3))
    annot  = np.empty((3, 3), dtype=object)
    for lc_v in range(3):
        for pc_v in range(3):
            s = stats(df_t[(df_t["lc"] == lc_v) & (df_t["pc"] == pc_v)])
            matrix[lc_v, pc_v] = s[metric]
            annot[lc_v, pc_v]  = f"{s[metric]:{fmt}}\n(N={s['n']})"
    vmax = np.abs(matrix).max()
    im = ax2[i].imshow(matrix, cmap="RdYlGn" if metric == "wr" else "RdYlGn",
                       aspect="auto", vmin=-vmax if metric == "avg" else 0, vmax=vmax if metric == "avg" else 100)
    ax2[i].set_xticks(range(3))
    ax2[i].set_yticks(range(3))
    ax2[i].set_xticklabels([pc_names[v] for v in range(3)], color="white", fontsize=9)
    ax2[i].set_yticklabels([lc_names[v] for v in range(3)], color="white", fontsize=9)
    ax2[i].set_xlabel("Pool Context", color="white")
    ax2[i].set_ylabel("London Context", color="white")
    ax2[i].set_title(title, color="white", fontsize=11)
    for lc_v in range(3):
        for pc_v in range(3):
            ax2[i].text(pc_v, lc_v, annot[lc_v, pc_v], ha="center", va="center",
                        fontsize=9, color="black", fontweight="bold")
    plt.colorbar(im, ax=ax2[i])

plt.tight_layout()
p2 = OUT_DIR / "0950_heatmap.png"
plt.savefig(p2, dpi=130, facecolor=fig2.get_facecolor())
plt.close()
print(f"[chart] {p2}")

# ── CHART 3 : Courbe equity + distribution P&L ───────────────────
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(14, 5))
fig3.patch.set_facecolor("#0d0d1a")
fig3.suptitle("Macro 09:50 — Equity & Distribution P&L (full dataset)", fontsize=12, fontweight="bold", color="white")

for ax in [ax3a, ax3b]:
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color("#444")

# Equity courbe
eq = [1.0]
for pnl in df_t["pnl"].values:
    eq.append(eq[-1] * (1 + pnl))
ax3a.plot(eq, color="#4CAF50", linewidth=1.5)
ax3a.fill_between(range(len(eq)), 1.0, eq, where=[e >= 1 for e in eq], alpha=0.2, color="#4CAF50")
ax3a.fill_between(range(len(eq)), 1.0, eq, where=[e < 1 for e in eq], alpha=0.2, color="#F44336")
ax3a.axhline(1.0, color="white", linewidth=0.8, linestyle="--", alpha=0.5)
ax3a.set_title("Courbe equity (ordre chronologique)", color="white", fontsize=10)
ax3a.set_xlabel("Trade #", color="white")
ax3a.set_ylabel("Equity", color="white")
final_ret = (eq[-1] - 1) * 100
ax3a.text(0.05, 0.92, f"Return : {final_ret:+.1f}%", transform=ax3a.transAxes,
          color="white", fontsize=10, fontweight="bold")

# Distribution P&L
pnls_pct = df_t["pnl"].values * 100
bins = np.linspace(pnls_pct.min() - 0.1, pnls_pct.max() + 0.1, 30)
wins   = pnls_pct[pnls_pct > 0]
losses = pnls_pct[pnls_pct < 0]
ax3b.hist(losses, bins=bins, color="#F44336", alpha=0.75, label=f"Losses (N={len(losses)})")
ax3b.hist(wins,   bins=bins, color="#4CAF50", alpha=0.75, label=f"Wins (N={len(wins)})")
ax3b.axvline(0, color="white", linewidth=1, linestyle="--", alpha=0.6)
ax3b.axvline(pnls_pct.mean(), color="#FF9800", linewidth=1.5, linestyle="-", label=f"Avg {pnls_pct.mean():+.3f}%")
ax3b.set_title("Distribution P&L", color="white", fontsize=10)
ax3b.set_xlabel("P&L (%)", color="white")
ax3b.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")

plt.tight_layout()
p3 = OUT_DIR / "0950_equity_dist.png"
plt.savefig(p3, dpi=130, facecolor=fig3.get_facecolor())
plt.close()
print(f"[chart] {p3}")

# ── CHART 4 : Performance par annee + mois ───────────────────────
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 5))
fig4.patch.set_facecolor("#0d0d1a")
fig4.suptitle("Macro 09:50 — Performance temporelle", fontsize=12, fontweight="bold", color="white")

for ax in [ax4a, ax4b]:
    ax.set_facecolor("#1a1a2e")
    ax.tick_params(colors="white")
    for sp in ["top","right"]: ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color("#444")

# Par annee
years = sorted(df_t["year"].unique())
yr_wr  = [stats(df_t[df_t["year"]==y])["wr"]  for y in years]
yr_avg = [stats(df_t[df_t["year"]==y])["avg"] for y in years]
yr_n   = [stats(df_t[df_t["year"]==y])["n"]   for y in years]
x = np.arange(len(years))
ax4a.bar(x - 0.2, yr_wr,  0.38, color=[color_wr(w) for w in yr_wr],  alpha=0.85, label="WR%")
ax4a.bar(x + 0.2, yr_avg, 0.38, color=["#4CAF50" if a > 0 else "#F44336" for a in yr_avg], alpha=0.85, label="Avg%")
ax4a.axhline(0, color="white", linewidth=0.5, alpha=0.3)
ax4a.set_xticks(x)
ax4a.set_xticklabels([f"{y}\n(N={n})" for y, n in zip(years, yr_n)], color="white", fontsize=9)
ax4a.set_title("Par annee", color="white", fontsize=10)
ax4a.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")

# Par mois (1-12)
months = range(1, 13)
mo_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
mo_wr  = [stats(df_t[df_t["month"]==m])["wr"]  for m in months]
mo_avg = [stats(df_t[df_t["month"]==m])["avg"] for m in months]
mo_n   = [stats(df_t[df_t["month"]==m])["n"]   for m in months]
x2 = np.arange(12)
ax4b.bar(x2 - 0.2, mo_wr,  0.38, color=[color_wr(w) for w in mo_wr],  alpha=0.85, label="WR%")
ax4b.bar(x2 + 0.2, mo_avg, 0.38, color=["#4CAF50" if a > 0 else "#F44336" for a in mo_avg], alpha=0.85, label="Avg%")
ax4b.axhline(0, color="white", linewidth=0.5, alpha=0.3)
ax4b.set_xticks(x2)
ax4b.set_xticklabels([f"{l}\n({n})" for l, n in zip(mo_labels, mo_n)], color="white", fontsize=7)
ax4b.set_title("Par mois calendaire", color="white", fontsize=10)
ax4b.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="white")

plt.tight_layout()
p4 = OUT_DIR / "0950_temporel.png"
plt.savefig(p4, dpi=130, facecolor=fig4.get_facecolor())
plt.close()
print(f"[chart] {p4}")

print("\n[analyse] Termine.")
