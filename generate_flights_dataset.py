# -*- coding: utf-8 -*-
"""Generate the flight datasets used by Data with Roots - R2A2 (Unsupervised Learning).

1) data/flights_dataset.csv       -> K-Means Clustering + Binary target (1,000 records)
     X: distance_km, duration_min | Y: delayed (0 = On time, 1 = Delayed)
2) data/flights_manual_100.csv    -> Manual K-Means simulation (100 records)
     X: distance_km, duration_min
"""
import os
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

rng = np.random.default_rng(42)
n = 1100

n_short = 420
n_med = 400
n_long = 280

short = rng.normal(loc=420, scale=110, size=n_short)
medium = rng.normal(loc=1450, scale=260, size=n_med)
longh = rng.normal(loc=2600, scale=330, size=n_long)

distance = np.concatenate([short, medium, longh])
distance = np.clip(distance, 100, 3200)

duration = 20 + distance * 0.082 + rng.normal(scale=14, size=n)

p_short = 0.18
p_med = 0.55
p_long = 0.85
p_delay = np.where(
    distance < 900, p_short,
    np.where(distance < 2000, p_med, p_long),
)
delayed_mask = rng.random(n) < p_delay

delay_time = np.where(
    delayed_mask,
    rng.gamma(shape=3.2, scale=20, size=n) + distance * 0.05,
    rng.normal(loc=2, scale=5, size=n),
)

df = pd.DataFrame({
    "flight_id": np.arange(1, n + 1),
    "distance_km": np.round(distance, 1),
    "duration_min": np.round(duration, 1),
    "delay_min": np.round(delay_time, 1),
})
df["delayed"] = (df["delay_min"] > 15).astype(int)

# -------- 1,000 records for the clustering + classification application --------
n_main = 1000
main = df.iloc[:n_main].copy().drop(columns=["delay_min"])
main.to_csv(os.path.join(DATA_DIR, "flights_dataset.csv"), index=False)

# -------- 100 records for the manual K-Means exercise (3 natural groups) ---------
rng2 = np.random.default_rng(7)
small = rng2.normal(loc=430, scale=80, size=34)
med = rng2.normal(loc=1480, scale=180, size=34)
longd = rng2.normal(loc=2650, scale=250, size=32)
manual_dist = np.clip(np.concatenate([small, med, longd]), 150, 3200)
manual_dur = 20 + manual_dist * 0.082 + rng2.normal(scale=12, size=100)
manual = pd.DataFrame({
    "flight_id": np.arange(n_main + 1, n_main + 101),
    "distance_km": np.round(manual_dist, 1),
    "duration_min": np.round(manual_dur, 1),
})
manual.to_csv(os.path.join(DATA_DIR, "flights_manual_100.csv"), index=False)

print("=== flights_dataset.csv ===")
print(main[["flight_id", "distance_km", "duration_min", "delayed"]].head(3))
print("records:", len(main), "| delay rate:", round(main["delayed"].mean(), 4))
print("=== flights_manual_100.csv ===")
print(manual.head(3))
print("records:", len(manual))