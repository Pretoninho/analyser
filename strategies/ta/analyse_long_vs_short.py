import sys, pandas as pd, numpy as np
sys.path.insert(0, ".")

is_df  = pd.read_csv("strategies/ta/results/sweep_filtered_IS.csv")
oos_df = pd.read_csv("strategies/ta/results/sweep_IS_vs_OOS.csv")
full   = pd.read_csv("strategies/ta/results/sweep_full_IS.csv")

# ── 1. Stats globales LONG vs SHORT (IS) ────────────────────────────────────
print("=== 1. LONG vs SHORT -- IS filtered (n>=20, WR>=55%, Exp>=0.10) ===")
for d in ["LONG","SHORT"]:
    sub = is_df[is_df.direction==d]
    print(f"  {d:5s}  n_etats={len(sub):4d}  WR_moy={sub.wr.mean():.3f}  Exp_moy={sub.exp_R.mean():.3f}  "
          f"WR_max={sub.wr.max():.3f}  Exp_max={sub.exp_R.max():.3f}")
print()

# ── 2. LONG vs SHORT dans le full (TOUS les etats, sans filtre) ──────────────
print("=== 2. LONG vs SHORT -- FULL sans filtre ===")
for d in ["LONG","SHORT"]:
    sub = full[full.direction==d]
    print(f"  {d:5s}  n_etats={len(sub):5d}  WR_moy={sub.wr.mean():.3f}  Exp_moy={sub.exp_R.mean():.3f}  "
          f"n_trades_moy={sub.n.mean():.1f}")
print()

# ── 3. Taux de passage filtre par direction ──────────────────────────────────
print("=== 3. Taux de passage du filtre (n>=20, WR>=55%, Exp>=0.10) ===")
for d in ["LONG","SHORT"]:
    tot   = len(full[full.direction==d])
    filt  = len(is_df[is_df.direction==d])
    print(f"  {d:5s}  total={tot:5d}  valides={filt:4d}  taux={filt/tot:.1%}")
print()

# ── 4. Distribution WR LONG vs SHORT (full) ─────────────────────────────────
print("=== 4. Distribution WR (full, tous états) ===")
bins   = [0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.01]
labels = ["<40%","40-45%","45-50%","50-55%","55-60%","60-65%","65-70%","70-75%","75%+"]
for d in ["LONG","SHORT"]:
    sub = full[full.direction==d].copy()
    sub["wr_bin"] = pd.cut(sub.wr, bins=bins, labels=labels, right=False)
    print(f"  {d}:")
    print("  " + sub.wr_bin.value_counts().sort_index().to_string().replace("\n","\n  "))
    print()

# ── 5. WR LONG par combinaison de features (full, n>=10) ────────────────────
print("=== 5. WR moyen LONG par etat structurel (full, n>=10) ===")
long_full = full[(full.direction=="LONG") & (full.n>=10)]
print("  Par ema_state:")
print("  " + long_full.groupby("ema_state").wr.mean().round(3).to_string())
print("  Par swing:")
print("  " + long_full.groupby("swing").wr.mean().round(3).to_string())
print("  Par rsi_state:")
print("  " + long_full.groupby("rsi_state").wr.mean().round(3).to_string())
print("  Par atr_state:")
print("  " + long_full.groupby("atr_state").wr.mean().round(3).to_string())
print("  Par vwap_state:")
print("  " + long_full.groupby("vwap_state").wr.mean().round(3).to_string())
print()

# ── 6. Meilleurs etats LONG valides en IS (top 20 par exp_R) ─────────────────
print("=== 6. Top 20 LONG IS par exp_R ===")
top_long = is_df[is_df.direction=="LONG"].sort_values("exp_R", ascending=False).head(20)
cols = ["params","ema_state","ema_slope","swing","rsi_state","stoch_state","atr_state","vwap_state","n","wr","exp_R"]
print(top_long[cols].to_string(index=False))
print()

# ── 7. Stabilite OOS des LONG vs SHORT ──────────────────────────────────────
print("=== 7. Stabilite OOS (wr_drop) LONG vs SHORT ===")
oos_df["wr_drop"] = oos_df["wr_OOS"] - oos_df["wr_IS"]
for d in ["LONG","SHORT"]:
    sub = oos_df[(oos_df.direction==d) & oos_df.wr_drop.notna()]
    stable   = (sub.wr_drop >= -0.05).sum()
    fragile  = (sub.wr_drop < -0.15).sum()
    print(f"  {d:5s}  n={len(sub):4d}  "
          f"stable(<=5% drop)={stable:3d} ({stable/len(sub):.1%})  "
          f"fragile(>15% drop)={fragile:3d} ({fragile/len(sub):.1%})  "
          f"wr_drop_moy={sub.wr_drop.mean():.3f}")
print()

# ── 8. Frequence des triggers LONG vs SHORT par session ──────────────────────
print("=== 8. Top LONG stables en OOS (drop<=5%, n_OOS>=5) ===")
solid_long = oos_df[
    (oos_df.direction=="LONG") &
    (oos_df.wr_drop.notna()) &
    (oos_df.wr_drop >= -0.05) &
    (oos_df.n_OOS >= 5)
].sort_values("exp_R_OOS", ascending=False)
oos_cols = ["params","ema_state","swing","rsi_state","stoch_state","atr_state",
            "n_IS","wr_IS","n_OOS","wr_OOS","wr_drop"]
print(solid_long.head(15)[oos_cols].to_string(index=False))
