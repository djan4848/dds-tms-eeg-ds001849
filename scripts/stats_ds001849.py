
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:     stats_ds001849.py
Purpose:   Group-level statistical analysis of DDS model parameters from OpenNeuro ds001849.
           Applies linear mixed-effects modeling (or OLS with fixed effects and cluster-robust SE)
           to test for site and condition effects on TMS–EEG parameter estimates.

Author:    Damian Jan (damian.jan@dejsl.com)
Affiliation:
           - Instituto Universitario de Neurociencias, Universidad de La Laguna, Spain
           - Dynamic Enterprise Junction Consulting S.L., Spain

Inputs:
    - derivatives/dds_ds001849/dds_params_group.csv:
        File with fitted DDS parameters (A1, f1, γ1, A2, f2, γ2, R2, RMSE) per subject × site × condition

Main functionalities:
    1. Aggregate DDS parameters by subject × site × condition using median
    2. Fit statistical models for each parameter:
         - Model: y ~ C(site) * C(cond) + (1 | subject)
         - Fallback: OLS + subject fixed effects + cluster-robust SE if MixedLM fails
    3. Apply Benjamini–Hochberg FDR correction per parameter
    4. Export:
         - Model summaries (`dds_group_stats.csv`)
         - Descriptive stats per site × condition (`dds_group_descriptives.csv`)
         - Boxplots per condition (`box_param_cond-*.png`)

Outputs:
    - dds_group_stats.csv: statistical effects (estimates, p-values, FDR) for each parameter
    - dds_group_descriptives.csv: group means, SDs, SEs, and 95% CIs per site × condition
    - figs/box_<param>_cond-<cond>.png: boxplots for publication (optional)

Dependencies:
    - Python ≥3.8
    - numpy, pandas, matplotlib
    - statsmodels

Usage:
    python stats_ds001849.py \
        --bids-root "/path/to/ds001849" \
        --params gamma1 f1 gamma2 f2 A1 A2 \
        --save-png

License:   MIT License 

Citation:
    Please cite the associated Journal of Neuroscience Methods article introducing the DDS model.

Notes:
    - MixedLM fitting uses `method="lbfgs"` with fallback to OLS when necessary
    - Boxplots are grouped by condition (Active vs Sham) and site (M1, DLPFC, PPC)
    - Input files should follow the BIDS derivatives structure used in this pipeline
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

def fdr_bh(p_vals, alpha=0.05):
    p = np.asarray(p_vals, dtype=float)
    n = p.size
    if n == 0:
        return np.array([], dtype=bool), np.array([])
    order = np.argsort(p)
    ranked = p[order]
    bh = ranked * n / (np.arange(n) + 1)
    bh = np.minimum.accumulate(bh[::-1])[::-1]
    inv = np.empty_like(order)
    inv[order] = np.arange(n)
    padj = bh[inv]
    reject = padj <= alpha
    return reject, padj

def aggregate(df):
    # Determinar site/cond: usar columnas explícitas si existen; si no, los 'guess'
    site = df["site"] if "site" in df.columns else df.get("site_guess")
    cond = df["cond"] if "cond" in df.columns else df.get("cond_guess")
    df = df.copy()
    df["site"] = site.astype(str)
    df["cond"] = cond.astype(str)
    keep = ["subject","site","cond","run","R2","RMSE",
            "A1","gamma1","f1","A2","gamma2","f2"]
    df = df[[c for c in keep if c in df.columns]].dropna(subset=["site","cond"])
    agg = df.groupby(["subject","site","cond"], dropna=False).median(numeric_only=True).reset_index()
    return agg

def mixedlm_or_fe(df_long):
    df_long = df_long.copy()
    df_long["subject"] = df_long["subject"].astype(str)
    df_long["site"] = df_long["site"].astype(str)
    df_long["cond"] = df_long["cond"].astype(str)
    # MixedLM
    try:
        md = smf.mixedlm("y ~ C(site)*C(cond)", data=df_long, groups=df_long["subject"])
        m = md.fit(reml=False, method="lbfgs", maxiter=500)
        table = pd.DataFrame({"term": m.fe_params.index, "estimate": m.fe_params.values,
                              "t": m.tvalues.values, "p": m.pvalues.values})
        model_type = "MixedLM"
        return table, model_type
    except Exception:
        # OLS + FE sujeto + SE clusterizadas por sujeto
        ols = smf.ols("y ~ C(site)*C(cond) + C(subject)", data=df_long).fit(
            cov_type="cluster", cov_kwds={"groups": df_long["subject"]}
        )
        table = pd.DataFrame({"term": ols.params.index, "estimate": ols.params.values,
                              "t": ols.tvalues.values, "p": ols.pvalues.values})
        model_type = "OLS+FE(cluster subj)"
        return table, model_type

def boxplots(df, param, outdir, save_png=False):
    for cond in sorted(df["cond"].unique()):
        sub = df[df["cond"]==cond]
        order = sorted(sub["site"].unique())
        data = [sub[sub["site"]==s][param].dropna().values for s in order]
        fig = plt.figure(figsize=(6,4))
        plt.boxplot(data, labels=order, showfliers=False)
        plt.xlabel("Site")
        plt.ylabel(param)
        plt.title(f"{param} by site – cond={cond}")
        plt.tight_layout()
        if save_png:
            fig.savefig(outdir / f"box_{param}_cond-{cond}.png", dpi=150)
        else:
            plt.show()
        plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root", type=str, required=True)
    ap.add_argument("--params", nargs="+", default=["gamma1","f1","gamma2","f2","A1","A2"])
    ap.add_argument("--save-png", action="store_true")
    args = ap.parse_args()

    bids_root = Path(args.bids_root)
    csv = bids_root / "derivatives" / "dds_ds001849" / "dds_params_group.csv"
    if not csv.exists():
        raise SystemExit(f"No existe: {csv}")
    df = pd.read_csv(csv)
    agg = aggregate(df)

    outdir = bids_root / "derivatives" / "dds_ds001849" / "figs"
    outdir.mkdir(parents=True, exist_ok=True)

    all_tables = []
    all_desc = []
    for param in args.params:
        if param not in agg.columns:
            continue
        df_long = agg[["subject","site","cond",param]].dropna().rename(columns={param:"y"})
        if df_long.empty:
            continue
        table, model = mixedlm_or_fe(df_long)
        table["param"] = param
        table["model"] = model
        all_tables.append(table)
        # descriptivos
        desc = df_long.groupby(["site","cond"]).agg(n=("y","size"), mean=("y","mean"), sd=("y","std")).reset_index()
        desc["se"] = desc["sd"] / np.sqrt(desc["n"].clip(lower=1))
        desc["ci95"] = 1.96 * desc["se"]
        desc["param"] = param
        all_desc.append(desc)
        # plots
        boxplots(agg, param, outdir, save_png=args.save_png)

    if all_tables:
        res = pd.concat(all_tables, ignore_index=True)
        # FDR por parámetro
        res["p_fdr"] = np.nan
        for p in res["param"].unique():
            idx = res["param"]==p
            rej,q = fdr_bh(res.loc[idx,"p"].values)
            res.loc[idx,"p_fdr"] = q
        res.to_csv(bids_root / "derivatives" / "dds_ds001849" / "dds_group_stats.csv", index=False)
    if all_desc:
        des = pd.concat(all_desc, ignore_index=True)
        des.to_csv(bids_root / "derivatives" / "dds_ds001849" / "dds_group_descriptives.csv", index=False)
    print("Done.")

if __name__ == "__main__":
    main()
