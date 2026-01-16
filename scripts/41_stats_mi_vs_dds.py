from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

INP = Path("outputs/features/dds_plus_mi.csv")
OUT = Path("outputs/stats/mi_vs_dds.csv")

FILTERS = [
    ("all_rows", lambda df: df),
    ("fit_valid_only", lambda df: df[df["fit_valid"] == True] if "fit_valid" in df.columns else df),
    ("no_fit_issues", lambda df: df[df["fit_issues"].isna()] if "fit_issues" in df.columns else df),
]

TARGETS = [
    ("early", "gamma1"),
    ("early", "A1"),
    ("early", "RMSE_uV"),
    ("late", "gamma2"),
    ("late", "f2"),
    ("late", "RMSE_uV"),
]

def corr_stats(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    n = len(x)
    if n < 8:
        return dict(n=n, pearson_r=np.nan, pearson_p=np.nan, spearman_r=np.nan, spearman_p=np.nan)
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return dict(n=n, pearson_r=float(pr), pearson_p=float(pp), spearman_r=float(sr), spearman_p=float(sp))

def main():
    m = pd.read_csv(INP)
    if "mi_early_late_bits" not in m.columns:
        raise RuntimeError("mi_early_late_bits not found")

    rows = []
    for w, xcol in TARGETS:
        if xcol not in m.columns:
            continue
        dfw = m[m["window"] == w].copy()
        if dfw.empty:
            continue
        for fname, f in FILTERS:
            df = f(dfw)
            st = corr_stats(df[xcol], df["mi_early_late_bits"])
            rows.append({
                "window": w,
                "x": xcol,
                "y": "mi_early_late_bits",
                "filter": fname,
                **st,
                "x_mean": float(np.nanmean(df[xcol])),
                "y_mean": float(np.nanmean(df["mi_early_late_bits"])),
            })

    out = pd.DataFrame(rows).sort_values(["window","x","filter"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print("[OK] wrote:", OUT)
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
