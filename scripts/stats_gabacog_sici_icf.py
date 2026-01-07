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

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# -------------------------
# CONFIG
# -------------------------
DERIV = Path("derivatives/dds_gabacog")
INFILE = DERIV / "dds_params_channelwise.csv"
OUTDIR = DERIV / "stats"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Key parameters + optional full set
PARAMETERS = ["gamma1", "f2", "gamma2", "A1", "A2", "f1"]  # keep f2 for ICF signature
R2_THRESHOLD = 0.9   # robustness analysis

MIN_N = 5  # minimum subjects per group at channel-level

# -------------------------
# HELPERS
# -------------------------
def cohens_d_independent(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
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
    reject, p_fdr, _, _ = multipletests(df[p_col].values, method="fdr_bh")
    df = df.copy()
    df["p_fdr"] = p_fdr
    df["significant"] = reject
    return df

# -------------------------
# LOAD DATA
# -------------------------
df = pd.read_csv(INFILE)

# keep only valid groups
df = df[df["group"].isin(["CTL", "TOC"])].copy()

# sanity: ensure per-protocol windows behave as expected (non-fatal)
# (SICI likely has 10_100ms; ICF likely has 10_200ms)
# print(df.groupby(["protocol","window"])["subject_id"].nunique())

# -------------------------
# MAIN ANALYSIS FUNCTION
# -------------------------
def run_between_only(df_in: pd.DataFrame, robustness_label: str) -> pd.DataFrame:
    rows = []

    # Loop by protocol first (avoids window confusion across protocols)
    for protocol in sorted(df_in["protocol"].unique()):
        dfp = df_in[df_in["protocol"] == protocol]

        for window in sorted(dfp["window"].unique()):
            dfw = dfp[dfp["window"] == window]

            for param in PARAMETERS:
                # Channel-wise between-groups
                for ch in sorted(dfw["channel"].unique()):
                    x = dfw[(dfw["group"]=="CTL") & (dfw["channel"]==ch)][param].values
                    y = dfw[(dfw["group"]=="TOC") & (dfw["channel"]==ch)][param].values

                    if len(x) < MIN_N or len(y) < MIN_N:
                        continue

                    t, p = ttest_ind(x, y, equal_var=False, nan_policy="omit")
                    d = cohens_d_independent(x, y)

                    rows.append(dict(
                        comparison="between",
                        robustness=robustness_label,
                        protocol=protocol,
                        window=window,
                        channel=ch,
                        parameter=param,
                        n_ctl=int(np.isfinite(x).sum()),
                        n_toc=int(np.isfinite(y).sum()),
                        t=float(t),
                        p=float(p),
                        effect_size=float(d) if np.isfinite(d) else np.nan,
                    ))

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    # -------------------------
    # FDR correction
    # -------------------------
    out = []
    for (protocol, parameter, window, robustness), sub in res.groupby(["protocol","parameter","window","robustness"]):
        out.append(bh_fdr(sub))

    return pd.concat(out, ignore_index=True)

# -------------------------
# RUN ROBUSTNESS SETS
# -------------------------
results_all = run_between_only(df, robustness_label="all")
results_hq  = run_between_only(df[df["R2"] > R2_THRESHOLD], robustness_label=f"R2_gt_{R2_THRESHOLD}")

results = pd.concat([results_all, results_hq], ignore_index=True)

# -------------------------
# SAVE
# -------------------------
outfile = OUTDIR / "stats_dds_gabacog_between_channelwise.csv"
results.to_csv(outfile, index=False)

print("Saved:", outfile)
print("Summary (significant, FDR<0.05):")
print(results.groupby(["protocol","robustness","parameter"])["significant"].sum().sort_index())

