import sys, pandas as pd
sys.path.insert(0, ".")
from strategies.ta.features import STATE_COLS

oos = pd.read_csv("strategies/ta/results/sweep_IS_vs_OOS.csv")
oos["wr_drop"] = oos["wr_OOS"] - oos["wr_IS"]

print("Colonnes disponibles:")
print([c for c in oos.columns])
print()

# Stats globales avec regime
print("=== ETATS VALIDES IS PAR REGIME ===")
filt = pd.read_csv("strategies/ta/results/sweep_filtered_IS.csv")
if "regime" in filt.columns:
    print(filt.groupby(["regime","direction"])[["wr","exp_R","n"]].agg(
        {"wr":"mean","exp_R":"mean","n":["mean","count"]}
    ).round(3).to_string())
else:
    print("Colonne regime absente du filtered_IS")
print()

# Stats OOS par regime
print("=== STABILITE OOS PAR REGIME ===")
if "regime" in oos.columns:
    for reg in ["bull","bear","range"]:
        s = oos[(oos.regime==reg) & oos.wr_drop.notna()]
        if len(s)==0: continue
        stable  = (s.wr_drop >= -0.05).sum()
        fragile = (s.wr_drop < -0.15).sum()
        impr    = (s.wr_drop > 0).sum()
        print(f"  {reg:5s}  n={len(s):4d}  "
              f"stable={stable:3d}({stable/len(s):.1%})  "
              f"ameliore={impr:3d}({impr/len(s):.1%})  "
              f"fragile={fragile:3d}({fragile/len(s):.1%})  "
              f"wr_drop_moy={s.wr_drop.mean():.3f}")
else:
    print("Colonne regime absente du IS_vs_OOS")
print()

# Top configs par regime en OOS
print("=== TOP 10 PAR REGIME (exp_OOS, n_OOS>=5, drop<=-5%) ===")
if "regime" in oos.columns:
    solid = oos[(oos.wr_drop >= -0.05) & (oos.n_OOS >= 5)]
    cols = ["regime","params","direction","swing","rsi_state","stoch_state","atr_state",
            "n_IS","wr_IS","n_OOS","wr_OOS","exp_R_OOS","wr_drop"]
    cols = [c for c in cols if c in oos.columns]
    for reg in ["bull","bear","range"]:
        sub = solid[solid.regime==reg].sort_values("exp_R_OOS" if "exp_R_OOS" in solid.columns
                                                    else "wr_OOS", ascending=False).head(10)
        print(f"  -- {reg.upper()} --")
        if len(sub):
            print(sub[cols].to_string(index=False))
        else:
            print("  Aucun etat stable")
        print()
