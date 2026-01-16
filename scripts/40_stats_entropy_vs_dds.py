"""
40_stats_entropy_vs_dds.py

Valida hipótesis informacionales básicas:
- Correlación entre parámetros DDS y entropía en la misma ventana.

Entrada:
- outputs/features/dds_plus_entropy.csv

Salida:
- outputs/stats/entropy_vs_dds.csv
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

INP = Path("outputs/features/dds_plus_entropy.csv")
OUT = Path("outputs/stats/entropy_vs_dds.csv")

TARGETS = [
    # (window, x, y)
    ("early", "gamma1", "entropy_shannon_bits"),
    ("late",  "gamma1", "entropy_shannon_bits"),
    # si existen, las probamos también:
    ("late",  "gamma2", "entropy_shannon_bits"),
    ("late",  "f2",     "entropy_shannon_bits"),
]

FILTERS = [
    ("all_rows", lambda df: df),
    ("fit_valid_only", lambda df: df[df["fit_valid"] == True] if "fit_valid" in df.columns else df),
    ("no_fit_issues", lambda df: df[df["fit_issues"].isna()] if "fit_issues" in df.columns else df),
]

def corr_stats(x: np.ndarray, y: np.ndarray):
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    n = len(x)
    if n < 8:
        return dict(n=n, pearson_r=np.nan, pearson_p=np.nan, spearman_r=np.nan, spearman_p=np.nan)
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return dict(n=n, pearson_r=float(pr), pearson_p=float(pp), spearman_r=float(sr), spearman_p=float(sp))

def main():
    if not INP.exists():
        raise FileNotFoundError(INP)

    m = pd.read_csv(INP)

    rows = []
    for w, xcol, ycol in TARGETS:
        if xcol not in m.columns or ycol not in m.columns:
            continue

        mw = m[m["window"] == w].copy()
        if mw.empty:
            continue

        for fname, f in FILTERS:
            df = f(mw)
            if df.empty:
                continue
            st = corr_stats(df[xcol].to_numpy(float), df[ycol].to_numpy(float))
            rows.append({
                "window": w,
                "x": xcol,
                "y": ycol,
                "filter": fname,
                **st,
                "x_mean": float(np.nanmean(df[xcol])),
                "y_mean": float(np.nanmean(df[ycol])),
            })

    out = pd.DataFrame(rows).sort_values(["window", "x", "filter"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"[OK] wrote: {OUT}")
    print(out)

if __name__ == "__main__":
    main()

