#!/usr/bin/env python3
"""
Statistical analysis for DDS parameters in GABACOG SICI/ICF.

GOAL (final):
- NO SICI vs ICF inferential comparisons.
- Between-group ONLY: CTL vs TOC within each protocol.
- Channel-wise primary, ROI secondary (handled in a separate script if desired).
- Robustness:
    * all fits
    * high-quality fits only (R2 > 0.9)
- Multiple-comparison correction: BH-FDR per (protocol × parameter × window) within each robustness set.

Outputs:
- CSV with raw p, FDR-corrected p, effect sizes.
"""

#!/usr/bin/env python3
"""
ROI-level statistics for DDS parameters in GABACOG SICI/ICF.

GOAL:
- Between-group ONLY: CTL vs TOC within each protocol.
- No SICI vs ICF comparison.
- Focus parameters:
    * SICI  -> gamma1
    * ICF   -> f2
- Robustness:
    * all fits
    * R2 > 0.9
- Multiple-comparison correction: BH-FDR per protocol × parameter × window.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# -------------------------
# CONFIG
# -------------------------
DERIV = Path("derivatives/dds_gabacog")
INFILE = DERIV / "dds_params_roi.csv"
OUTDIR = DERIV / "stats"
OUTDIR.mkdir(parents=True, exist_ok=True)

PARAMS_BY_PROTOCOL = {
    "SICI": ["gamma1"],
    "ICF":  ["f2"],
}

R2_THRESHOLD = 0.9
MIN_N = 5

# -------------------------
# HELPERS
# -------------------------
def cohens_d(x, y):
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    if pooled == 0:
        return np.nan
    return (np.mean(x) - np.mean(y)) / pooled

def bh_fdr(df, p_col="p"):
    reject, p_fdr, *_ = multipletests(df[p_col].values, method="fdr_bh")
    df = df.copy()
    df["p_fdr"] = p_fdr
    df["significant"] = reject
    return df

# -------------------------
# LOAD
# -------------------------
df = pd.read_csv(INFILE)
df = df[df.group.isin(["CTL","TOC"])].copy()

# -------------------------
# MAIN
# -------------------------
rows = []

for robustness, dfi in [
    ("all", df),
    (f"R2_gt_{R2_THRESHOLD}", df[df.R2 > R2_THRESHOLD]),
]:
    for protocol, params in PARAMS_BY_PROTOCOL.items():
        dfp = dfi[dfi.protocol == protocol]

        for window in dfp.window.unique():
            dfw = dfp[dfp.window == window]

            for roi in dfw.roi.unique():
                for param in params:
                    x = dfw[(dfw.group=="CTL") & (dfw.roi==roi)][param].values
                    y = dfw[(dfw.group=="TOC") & (dfw.roi==roi)][param].values

                    if len(x) < MIN_N or len(y) < MIN_N:
                        continue

                    t, p = ttest_ind(x, y, equal_var=False, nan_policy="omit")
                    d = cohens_d(x, y)

                    rows.append(dict(
                        comparison="between",
                        robustness=robustness,
                        protocol=protocol,
                        window=window,
                        roi=roi,
                        parameter=param,
                        n_ctl=len(x),
                        n_toc=len(y),
                        t=t,
                        p=p,
                        effect_size=d,
                    ))

res = pd.DataFrame(rows)

# -------------------------
# FDR
# -------------------------
out = []
for keys, sub in res.groupby(["protocol","parameter","window","robustness"]):
    out.append(bh_fdr(sub))

res = pd.concat(out, ignore_index=True)

# -------------------------
# SAVE
# -------------------------
outfile = OUTDIR / "stats_dds_gabacog_between_roi.csv"
res.to_csv(outfile, index=False)

print("Saved:", outfile)
print("\nSummary (significant, FDR<0.05):")
print(res.groupby(["protocol","robustness","parameter"])["significant"].sum())
print("\nMin p / p_fdr:")
print(res.groupby(["protocol","robustness","parameter"])[["p","p_fdr"]].min())

