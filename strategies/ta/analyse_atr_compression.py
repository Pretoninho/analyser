import sys, pandas as pd, numpy as np
sys.path.insert(0, ".")

from strategies.ta.features import load_15m, _atr
from strategies.ta.config import ATR_BINS, ATR_LABELS

full   = pd.read_csv("strategies/ta/results/sweep_full_IS.csv")
oos    = pd.read_csv("strategies/ta/results/sweep_IS_vs_OOS.csv")
oos["wr_drop"] = oos["wr_OOS"] - oos["wr_IS"]

# ── 1. Stats ATR compression IS vs OOS ──────────────────────────────────────
print("=== 1. Performance IS par atr_state (full, n>=10) ===")
sub = full[full.n >= 10]
print(sub.groupby(["atr_state","direction"])[["wr","n"]].agg({"wr":"mean","n":"mean"}).round(3).to_string())
print()

print("=== 2. Stabilite OOS par atr_state ===")
for atr in ["compression","neutral","expansion"]:
    s = oos[(oos.atr_state==atr) & oos.wr_drop.notna()]
    if len(s)==0: continue
    stable  = (s.wr_drop >= -0.05).sum()
    fragile = (s.wr_drop < -0.15).sum()
    print(f"  {atr:12s}  n={len(s):4d}  stable={stable:3d}({stable/len(s):.1%})  "
          f"fragile={fragile:3d}({fragile/len(s):.1%})  wr_drop_moy={s.wr_drop.mean():.3f}")
print()

# ── 2. Evolution temporelle ATR compression dans les données brutes ──────────
print("=== 3. Frequence ATR compression par annee (donnees 15m) ===")
df15 = load_15m()
atr14 = _atr(df15["high"], df15["low"], df15["close"], 14)
atr_ma = atr14.rolling(20).mean()
atr_ratio = atr14 / atr_ma.replace(0.0, float("nan"))
atr_state = pd.cut(atr_ratio, bins=ATR_BINS, labels=ATR_LABELS, right=False)

yearly = pd.DataFrame({"atr_state": atr_state, "year": df15.index.year})
pivot = yearly.groupby(["year","atr_state"]).size().unstack(fill_value=0)
pivot["total"] = pivot.sum(axis=1)
for col in ["compression","neutral","expansion"]:
    if col in pivot.columns:
        pivot[f"{col}_%"] = (pivot[col] / pivot["total"] * 100).round(1)
print(pivot[[c for c in pivot.columns if "%" in c]].to_string())
print()

# ── 3. Valeur mediane de ATR ratio en compression par annee ─────────────────
print("=== 4. ATR ratio median par annee (compression seulement) ===")
df_atr = pd.DataFrame({"ratio": atr_ratio, "year": df15.index.year})
df_atr_comp = df_atr[atr_ratio < 0.8]
print(df_atr_comp.groupby("year")["ratio"].agg(["median","mean","count"]).round(3).to_string())
print()

# ── 4. Duree moyenne des phases de compression par annee ────────────────────
print("=== 5. Duree des phases de compression continues (en bougies 15m) ===")
is_comp = (atr_state == "compression").astype(int)
# groupby consecutive runs
runs = []
in_run = False
length = 0
for v in is_comp.values:
    if v == 1:
        in_run = True
        length += 1
    else:
        if in_run:
            runs.append(length)
        in_run = False
        length = 0

runs = np.array(runs)
print(f"  Nombre de phases   : {len(runs)}")
print(f"  Duree moyenne      : {runs.mean():.1f} bougies = {runs.mean()*15/60:.1f}h")
print(f"  Duree mediane      : {np.median(runs):.1f} bougies = {np.median(runs)*15/60:.1f}h")
print(f"  Duree max          : {runs.max():.0f} bougies = {runs.max()*15/60:.1f}h")
print(f"  Phases < 4 bougies : {(runs<4).sum()} ({(runs<4).mean():.1%})")
print(f"  Phases > 48 bougies: {(runs>48).sum()} ({(runs>48).mean():.1%})")
print()

# ── 5. IS vs OOS : proportion de bougies en compression ─────────────────────
print("=== 6. Proportion de bougies en compression IS (2020-2024) vs OOS (2025-2026) ===")
df_yearly = pd.DataFrame({
    "state": atr_state.values,
    "ts": df15.index
})
df_yearly["period"] = np.where(df_yearly["ts"] < pd.Timestamp("2025-01-01", tz="UTC"), "IS (2020-24)", "OOS (2025-26)")
pivot2 = df_yearly.groupby(["period","state"]).size().unstack(fill_value=0)
pivot2["total"] = pivot2.sum(axis=1)
for col in ["compression","neutral","expansion"]:
    if col in pivot2.columns:
        pivot2[f"{col}_%"] = (pivot2[col] / pivot2["total"] * 100).round(1)
print(pivot2[[c for c in pivot2.columns if "%" in c]].to_string())
print()

# ── 6. Analyse des etats IS compression fragiles ─────────────────────────────
print("=== 7. Etats compression fragiles OOS : patterns communs ===")
frag_comp = oos[
    (oos.atr_state=="compression") &
    (oos.wr_drop.notna()) &
    (oos.wr_drop < -0.15)
]
print(f"  Nombre d'etats fragiles compression : {len(frag_comp)}")
print("  Par direction:")
print("  " + frag_comp.direction.value_counts().to_string())
print("  Par rsi_state:")
print("  " + frag_comp.rsi_state.value_counts().to_string())
print("  Par swing:")
print("  " + frag_comp.swing.value_counts().to_string())
print("  Par stoch_state:")
print("  " + frag_comp.stoch_state.value_counts().to_string())
print()

# ── 7. Etats compression stables OOS (s'il y en a) ───────────────────────────
print("=== 8. Etats compression STABLES OOS (drop<=5%, n_OOS>=5) ===")
stab_comp = oos[
    (oos.atr_state=="compression") &
    (oos.wr_drop.notna()) &
    (oos.wr_drop >= -0.05) &
    (oos.n_OOS >= 5)
].sort_values("wr_OOS", ascending=False)
cols = ["params","direction","swing","rsi_state","stoch_state","n_IS","wr_IS","n_OOS","wr_OOS","wr_drop"]
print(stab_comp.head(15)[cols].to_string(index=False))
