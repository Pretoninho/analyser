import sys, pandas as pd
sys.path.insert(0, ".")

df = pd.read_csv("strategies/ta/results/sweep_filtered_IS.csv")

cols_disp = ["params","direction","ema_state","ema_slope","swing",
             "rsi_state","stoch_state","atr_state","vwap_state","n","wr","exp_R"]

print("=== STATISTIQUES GENERALES ===")
print(f"Total etats valides : {len(df)}")
print(f"Direction LONG  : {(df['direction']=='LONG').sum()}")
print(f"Direction SHORT : {(df['direction']=='SHORT').sum()}")
print(f"WR moyen        : {df['wr'].mean():.1%}")
print(f"Exp_R moyen     : {df['exp_R'].mean():.3f}R")
print(f"Exp_R max       : {df['exp_R'].max():.3f}R")
print()

print("=== TOP 25 PAR EXPECTANCY (exp_R) ===")
top = df.sort_values("exp_R", ascending=False).head(25)
print(top[cols_disp].to_string(index=False))
print()

print("=== WR MOYEN PAR EMA_STATE x DIRECTION ===")
print(df.groupby(["ema_state","direction"])[["wr","exp_R","n"]].mean().round(3).to_string())
print()

print("=== WR MOYEN PAR ATR_STATE x DIRECTION ===")
print(df.groupby(["atr_state","direction"])[["wr","exp_R","n"]].mean().round(3).to_string())
print()

print("=== WR MOYEN PAR RSI_STATE x DIRECTION ===")
print(df.groupby(["rsi_state","direction"])[["wr","exp_R","n"]].mean().round(3).to_string())
print()

print("=== WR MOYEN PAR SWING x DIRECTION ===")
print(df.groupby(["swing","direction"])[["wr","exp_R","n"]].mean().round(3).to_string())
print()

print("=== TOP 10 ETATS LES PLUS FREQUENTS (n trades) ===")
top_n = df.sort_values("n", ascending=False).head(10)
print(top_n[cols_disp].to_string(index=False))
print()

print("=== DISTRIBUTION WR (tous etats valides) ===")
bins = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1.01]
labels = ["55-60%","60-65%","65-70%","70-75%","75-80%","80%+"]
df["wr_bin"] = pd.cut(df["wr"], bins=bins, labels=labels, right=False)
print(df["wr_bin"].value_counts().sort_index().to_string())
