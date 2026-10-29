"""HSV oranını veri setindeki şiddet etiketleriyle karşılaştırır ve en iyi eşikleri bulur.
Çıktı: severity_thresholds.json + özet (Spearman korelasyonu, sınıf başına medyan oran)."""
import argparse, json

import cv2
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from severity import estimate_ratio, DEFAULT_THRESHOLDS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/val.csv")
    ap.add_argument("--per_level", type=int, default=300)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    df = df[df.severity >= 0]
    df = pd.concat([g.sample(min(len(g), a.per_level), random_state=0)
                    for _, g in df.groupby("severity")])
    ratios = []
    for p in tqdm(df.path):
        img = cv2.imread(p)
        ratios.append(estimate_ratio(img)[0] if img is not None else np.nan)
    df["ratio"] = ratios; df = df.dropna(subset=["ratio"])
    print("\nSeviye başına oran (%)  [medyan | 25-75 yüzdelik]")
    for s, g in df.groupby("severity"):
        print(f"  seviye {s}: {g.ratio.median():.1f} | {g.ratio.quantile(.25):.1f}-{g.ratio.quantile(.75):.1f}  (n={len(g)})")
    rho, _ = spearmanr(df.ratio, df.severity)
    print(f"Spearman korelasyonu (oran vs etiket): {rho:.3f}")

    best, grid = (0, DEFAULT_THRESHOLDS), np.arange(0.5, 60, 0.5)
    r, y = df.ratio.values, df.severity.values
    for t1 in grid:
        for t2 in grid[grid > t1]:
            acc = (np.where(r < t1, 0, np.where(r < t2, 1, 2)) == y).mean()
            if acc > best[0]:
                best = (acc, dict(t_general=float(t1), t_serious=float(t2)))
    print(f"En iyi eşikler: {best[1]}  -> doğruluk {best[0]:.3f}")
    json.dump(best[1], open("severity_thresholds.json", "w"), indent=2)
    df.to_csv("severity_calibration.csv", index=False)
    print("Kaydedildi: severity_thresholds.json, severity_calibration.csv")


if __name__ == "__main__":
    main()
