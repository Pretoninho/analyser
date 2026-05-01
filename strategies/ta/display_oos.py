import sys, pandas as pd
sys.path.insert(0, ".")

df = pd.read_csv("strategies/ta/results/sweep_IS_vs_OOS.csv")

# Renommer pour lisibilite
df = df.rename(columns={
    "n_IS": "n_IS", "wr_IS": "wr_IS", "exp_R_IS": "exp_IS",
    "n_OOS": "n_OOS", "wr_OOS": "wr_OOS", "exp_R_OOS": "exp_OOS",
})

# Calculer la degradation WR
df["wr_drop"] = (df["wr_OOS"] - df["wr_IS"]).round(4)
df["exp_drop"] = (df["exp_OOS"] - df["exp_IS"]).round(4)

total = len(df)
has_oos = df["n_OOS"].notna().sum()
stable = ((df["wr_drop"] >= -0.05) & df["n_OOS"].notna()).sum()
improved = ((df["wr_drop"] > 0) & df["n_OOS"].notna()).sum()

print("=== OOS GLOBAL ===")
print(f"Etats IS valides          : {total}")
print(f"Etats avec trades OOS     : {has_oos}")
print(f"Etats stables (WR drop <= 5%)   : {stable}")
print(f"Etats ameliores en OOS    : {improved}")
print()

cols = ["params","direction","ema_state","swing","rsi_state","stoch_state",
        "atr_state","n_IS","wr_IS","exp_IS","n_OOS","wr_OOS","exp_OOS","wr_drop"]

# Top IS qui tiennent en OOS (stables ou ameliores, n_OOS >= 5)
solid = df[(df["wr_drop"] >= -0.05) & (df["n_OOS"] >= 5)].sort_values("exp_OOS", ascending=False)
print(f"=== TOP 30 ETATS SOLIDES IS->OOS (WR drop <= 5%, n_OOS >= 5) ===")
print(solid.head(30)[cols].to_string(index=False))
print()

# Etats qui se degradent fortement
crash = df[(df["wr_drop"] < -0.15) & df["n_OOS"].notna()].sort_values("wr_drop")
print(f"=== ETATS FRAGILES (WR drop > 15%) : {len(crash)} ===")
print(crash.head(15)[cols].to_string(index=False))
print()

# Distribution degradation
print("=== DISTRIBUTION WR DROP (IS -> OOS) ===")
bins2 = [-1, -0.20, -0.10, -0.05, 0.0, 0.05, 1]
lab2  = ["<-20%", "-20 a -10%", "-10 a -5%", "-5 a 0%", "0 a +5%", ">+5%"]
df["drop_bin"] = pd.cut(df["wr_drop"], bins=bins2, labels=lab2, right=False)
print(df.dropna(subset=["wr_drop"])["drop_bin"].value_counts().sort_index().to_string())
print()

# Top IS par exp_R avec colonnes OOS cote a cote
print("=== TOP 20 IS PAR EXP_R -- COMPARAISON IS vs OOS ===")
top = df.sort_values("exp_IS", ascending=False).head(20)
print(top[cols].to_string(index=False))
