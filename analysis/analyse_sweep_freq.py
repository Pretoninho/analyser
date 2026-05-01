"""
Analyse la frequence de sweep a 09:50 ET sur les donnees historiques locales.
Diagnostic : combien de jours ont un sweep (sc!=0) par trimestre.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pytz
from data.binance import load_binance_1m

df = load_binance_1m()
ET = pytz.timezone("America/New_York")
df["ts_et"]   = df["timestamp"].dt.tz_convert(ET)
df["date_et"] = df["ts_et"].dt.date
df["hm_et"]   = df["ts_et"].dt.hour * 60 + df["ts_et"].dt.minute
df["dow"]     = df["ts_et"].dt.dayofweek

# 200 derniers jours de trading
dates = sorted(df["date_et"].unique())[-200:]
df = df[df["date_et"].isin(dates)]

results = []
for d, g in df.groupby("date_et"):
    dow = int(g["dow"].iloc[0])
    if dow == 0:
        continue
    pre   = g[(g["hm_et"] >= 570) & (g["hm_et"] < 590)]
    first = g[g["hm_et"] == 590]
    if len(pre) < 3 or first.empty:
        continue
    pre_h = float(pre["high"].max())
    pre_l = float(pre["low"].min())
    f = first.iloc[0]
    if float(f["high"]) > pre_h:
        sc = 1
    elif float(f["low"]) < pre_l:
        sc = 2
    else:
        sc = 0
    results.append({"date": pd.Timestamp(d), "dow": dow, "sc": sc})

r = pd.DataFrame(results)
print(f"Total jours analyses (hors lundi) : {len(r)}")
n_sweep = int((r["sc"] != 0).sum())
pct = n_sweep / len(r) * 100
print(f"Sweep sc!=0 : {n_sweep} ({pct:.1f}%)")
print(f"  SWEEP_H sc=1 : {int((r['sc']==1).sum())} ({(r['sc']==1).mean()*100:.1f}%)")
print(f"  SWEEP_L sc=2 : {int((r['sc']==2).sum())} ({(r['sc']==2).mean()*100:.1f}%)")
print(f"  NO_SWEEP sc=0 : {int((r['sc']==0).sum())} ({(r['sc']==0).mean()*100:.1f}%)")
print()

r["qtr"] = r["date"].dt.to_period("Q")
print("Frequence sweep par trimestre:")
for q, g in r.groupby("qtr"):
    n = len(g)
    ns = int((g["sc"] != 0).sum())
    print(f"  {q} : {ns}/{n} = {ns/n*100:.0f}%")

# 30 derniers jours
last30 = r[r["date"] >= r["date"].max() - pd.Timedelta(days=45)]
print()
print(f"30-45 derniers jours ({last30['date'].min().date()} -> {last30['date'].max().date()}):")
n30 = len(last30)
ns30 = int((last30["sc"] != 0).sum())
print(f"  Sweep : {ns30}/{n30} = {ns30/n30*100:.0f}%")
print()
print("Derniers 20 jours:")
for _, row in r.tail(20).iterrows():
    label = {0: "NO_SWEEP", 1: "SWEEP_H", 2: "SWEEP_L"}[row["sc"]]
    print(f"  {row['date'].date()} (dow={row['dow']})  {label}")
