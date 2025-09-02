#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:     compare_dds_vs_cosine.py
Purpose:   Compare the DDS parametric model with cosine similarity metrics (per Freedberg et al., 2020)
           on TMS–EEG data from OpenNeuro ds001849. Merges and analyzes both approaches to assess
           their complementarity in quantifying site- and condition-specific TEP dynamics.

Author:    Damian Jan (damian.jan@dejsl.com)
Affiliation:
           - Instituto Universitario de Neurociencias, Universidad de La Laguna, Spain
           - Dynamic Enterprise Junction Consulting S.L., Spain

Inputs:
    - cosine_similarity_results.csv: pairwise similarity per subject × condition × site pair
    - dds_params_group.csv: DDS parameters (A1, f1, γ1, etc.) for each TEP segment
    - Optionally: BIDS root directory

Main functionalities:
    1. Aggregate DDS parameters by subject × site × condition (median)
    2. Collapse cosine similarity to site-level by averaging pairs involving the same site
    3. Merge both datasets on subject × site × condition
    4. Run statistical models (OLS + subject fixed effects, cluster-robust SEs) for site/condition effects
    5. Compute Spearman correlations between cosine similarity and DDS parameters
    6. Generate journal-ready plots:
        - Paired boxplots (Active vs Sham)
        - Scatter plots
        - Correlation matrices

Outputs:
    - merged_dds_cosine_site.csv
    - stats_cond_site_FEcluster.csv
    - correlations_similarity_vs_dds.csv
    - correlation_matrix_phys_active_sham.png
    - paired_box_*.png / paired_scatter_*.png (if --paired-boxplots)
    - descriptives_by_site_cond.csv

Dependencies:
    - Python ≥3.8
    - numpy, pandas, matplotlib, seaborn
    - statsmodels, scipy

Usage:
    python compare_dds_vs_cosine.py \
        --bids-root "/path/to/ds001849" \
        --cosine-csv "/path/to/cosine_similarity_results.csv" \
        --dds-csv "/path/to/dds_params_group.csv" \
        --outdir "/path/to/output_directory" \
        --paired-boxplots \
        --save-png

License:   MIT License 

Citation:
    Please cite the original cosine similarity paper (Freedberg et al., 2020) and the associated
    Journal of Neuroscience Methods article describing the DDS model and this comparative analysis.

Notes:
    - Compatible with BIDS-structured EEG datasets
    - Assumes file naming encodes site and condition (e.g., DLPFC, sham)
    - All statistical models use cluster-robust standard errors at the subject level
"""



import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- High-quality, journal-ready defaults ----
mpl.rcParams.update({
    "figure.dpi": 150,               # para pantalla; el guardado usa dpi propio
    "savefig.dpi": 300,              # default si no se sobreescribe
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "font.family": "DejaVu Sans",    # o "Arial" si la tienes instalada
    "font.size": 10,                 # cuerpo base (ajústalo a 9–10 pt según revista)
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "axes.linewidth": 0.8,
    "grid.linestyle": ":",
    "grid.linewidth": 0.6,
})
# Tamaños recomendados de figura (columnas de revista)
FIGSIZE_1COL = (3.5, 3.0)   # ~1 columna (pulgadas)
FIGSIZE_2COL = (7.2, 4.5)   # ~2 columnas
# Parámetros de guardado centralizados
SAVE_KW = dict(dpi=600, bbox_inches="tight", pad_inches=0.01)  # PNG a 600 dpi

# ----------------- utilidades -----------------
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

def standardize_site(x: str) -> str:
    if not isinstance(x, str):
        return "unknown"
    s = x.lower()
    if "motor" in s or s == "m1":
        return "m1"
    if "frontal" in s or "dlpfc" in s or "pfc" in s:
        return "dlpfc"
    if "parietal" in s or "ppc" in s:
        return "ppc"
    return s

def standardize_cond(x: str) -> str:
    if not isinstance(x, str):
        return "unknown"
    s = x.lower()
    if "sham" in s:
        return "sham"
    if "active" in s or "real" in s or "tms" in s:
        return "active"
    return s

def mixed_or_fe(df_long, formula, cluster="subject"):
    """
    OLS + FE(subject) con SE clusterizadas por sujeto (robusto y estable).
    Devuelve DataFrame con términos, estimaciones y p-valores.
    """
    m = smf.ols(formula, data=df_long).fit(
        cov_type="cluster", cov_kwds={"groups": df_long[cluster].astype(str)}
    )
    table = pd.DataFrame({
        "term": m.params.index,
        "estimate": m.params.values,
        "t": m.tvalues.values,
        "p": m.pvalues.values
    })
    return table, m

# ----------------- agregados -----------------
def aggregate_dds(dds_df: pd.DataFrame) -> pd.DataFrame:
    # elegir columnas disponibles y estandarizar etiquetas
    df = dds_df.copy()
    # determinar site y cond preferentemente explícitos, si no *_guess
    site = df["site"] if "site" in df.columns else df.get("site_guess")
    cond = df["cond"] if "cond" in df.columns else df.get("cond_guess")
    df["site"] = site.astype(str).map(standardize_site)
    df["cond"] = cond.astype(str).map(standardize_cond)
    df["subject"] = df["subject"].astype(str)

    keep = ["subject","site","cond","run","R2","RMSE","A1","gamma1","f1","A2","gamma2","f2"]
    cols = [c for c in keep if c in df.columns]
    df = df[cols].dropna(subset=["site","cond"])

    # Mediana por sujeto×sitio×condición para robustez
    agg = df.groupby(["subject","site","cond"], dropna=False).median(numeric_only=True).reset_index()
    return agg

def collapse_cosine_to_site(cos_df: pd.DataFrame) -> pd.DataFrame:
    """
    A partir de filas con (subject, cond, site1, site2, similarity),
    construye una métrica de similarity por sitio: para cada (subject, cond, site)
    promedio de similitudes de las parejas en las que participa 'site'.
    """
    df = cos_df.copy()
    df["subject"] = df["subject"].astype(str)
    df["cond"] = df["cond"].astype(str).map(standardize_cond)
    df["site1"] = df["site1"].astype(str).map(standardize_site)
    df["site2"] = df["site2"].astype(str).map(standardize_site)

    rows = []
    for (subj, cond), sub in df.groupby(["subject","cond"]):
        # recolectar para cada sitio todas las similitudes en que participa
        for site in sorted(set(sub["site1"]).union(set(sub["site2"]))):
            sims = []
            for _, r in sub.iterrows():
                if r["site1"] == site or r["site2"] == site:
                    sims.append(r["similarity"])
            if len(sims):
                rows.append({
                    "subject": subj,
                    "cond": cond,
                    "site": site,
                    "similarity_site": float(np.nanmean(sims)),
                    "n_pairs": int(len(sims))
                })
    site_df = pd.DataFrame(rows)
    return site_df

# ----------------- gráficos -----------------
def boxplot_by_site(df, y, outdir, title, save_png=False):
    order = [s for s in ["m1","dlpfc","ppc"] if s in df["site"].unique()]
    for cond in sorted(df["cond"].unique()):
        sub = df[df["cond"] == cond]
        if not len(sub):
            continue
        data = [sub[sub["site"]==s][y].dropna().values for s in order]
        fig = plt.figure(figsize=(6,4))
        plt.boxplot(data, labels=order, showfliers=False)
        plt.xlabel("Sitio")
        plt.ylabel(y)
        plt.title(f"{title} – cond={cond}")
        plt.tight_layout()
        if save_png:
            fig.savefig(outdir / f"box_{y}_cond-{cond}.png", dpi=600)
        plt.close(fig)

def scatter_similarity_vs_param(merged, param, outdir, save_png=False):
    fig = plt.figure(figsize=(5,4))
    plt.scatter(merged["similarity_site"], merged[param], alpha=0.6)
    plt.xlabel("Cosine similarity (por sitio)")
    plt.ylabel(param)
    plt.title(f"{param} vs. cosine similarity")
    plt.tight_layout()
    if save_png:
        fig.savefig(outdir / f"scatter_{param}_vs_similarity.png", dpi=600)
    plt.close(fig)
# ----------------- plotting utilities (English, high-res, paired) -----------------
import math

def _site_order(df):
    return [s for s in ["m1","dlpfc","ppc"] if s in df["site"].unique()]

def _global_ylim_for_param(df, param, cond_vals=("active","sham")):
    vals = []
    for c in cond_vals:
        sub = df[df["cond"]==c]
        if not sub.empty and param in sub.columns:
            vals.append(sub[param].dropna().values)
    if not vals:
        return None
    v = np.concatenate(vals)
    if v.size == 0:
        return None
    lo, hi = np.nanmin(v), np.nanmax(v)
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    pad = 0.05 * (hi - lo if hi > lo else (abs(hi) + 1e-12))
    return (lo - pad, hi + pad)

def paired_boxplot_by_site(df, param, outdir, title=None, save=True):
    """
    Side-by-side boxplots (Active left, Sham right) with shared Y scale.
    - df: merged table (subject × site × cond)
    - param: column to plot (e.g., 'A1', 'gamma1', 'f1', 'R2', 'similarity_site', ...)
    """
    conds = ["active", "sham"]
    available = [c for c in conds if c in df["cond"].unique()]
    if len(available) == 0:
        return

    sites = _site_order(df)
    if len(sites) == 0:
        return

    ylim = _global_ylim_for_param(df, param, tuple(available))

    fig, axes = plt.subplots(1, len(available), figsize=(10, 4.5), sharey=True)  # high-res later on save
    if len(available) == 1:
        axes = [axes]

    for ax, cond in zip(axes, available):
        sub = df[(df["cond"] == cond) & (df[param].notna())]
        data = [sub.loc[sub["site"]==s, param].dropna().values for s in sites]
        ax.boxplot(data, labels=[s.upper() for s in sites], showfliers=False)
        ax.set_xlabel("Site")
        ax.set_title(f"{cond.capitalize()}")
        ax.grid(True, axis="y", linestyle=":", linewidth=0.6)
        if ylim is not None:
            ax.set_ylim(ylim)

    axes[0].set_ylabel(param)
    if title is None:
        title = f"{param} by site (Active vs Sham)"
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.00, 1, 0.95])

    if save:
        png = outdir / f"paired_box_{param}.png"
        fig.savefig(png, **SAVE_KW) # high quality PNG
    plt.close(fig)

def scatter_similarity_vs_param(merged, param, outdir, save_png=False):
    # (sin cambios sustanciales; etiquetas en inglés)
    fig = plt.figure(figsize=(5,4))
    plt.scatter(merged["similarity_site"], merged[param], alpha=0.6)
    plt.xlabel("Cosine similarity (per site)")
    plt.ylabel(param)
    plt.title(f"{param} vs cosine similarity")
    plt.tight_layout()
    if save_png:
        fig.savefig(outdir / f"scatter_{param}_vs_similarity.png", dpi=600)
       
    plt.close(fig)
def paired_scatter_similarity_vs_param(df, param, outdir, save=True):
    """
    Side-by-side scatterplots (Active left, Sham right) with shared axes.
    Shows cosine similarity vs a DDS parameter.
    """
    conds = ["active", "sham"]
    available = [c for c in conds if c in df["cond"].unique()]
    if len(available) < 1 or param not in df.columns:
        return

    # Determine global limits for same axes across both conditions
    x_vals = df["similarity_site"].dropna().values
    y_vals = df[param].dropna().values
    if x_vals.size == 0 or y_vals.size == 0:
        return

    x_min, x_max = np.nanmin(x_vals), np.nanmax(x_vals)
    y_min, y_max = np.nanmin(y_vals), np.nanmax(y_vals)
    # small padding
    dx = 0.05 * (x_max - x_min if x_max > x_min else abs(x_max))
    dy = 0.05 * (y_max - y_min if y_max > y_min else abs(y_max))
    xlim = (x_min - dx, x_max + dx)
    ylim = (y_min - dy, y_max + dy)

    fig, axes = plt.subplots(1, len(available), figsize=(10, 4.5), sharex=True, sharey=True)
    if len(available) == 1:
        axes = [axes]

    for ax, cond in zip(axes, available):
        sub = df[(df["cond"] == cond) & df[param].notna() & df["similarity_site"].notna()]
        if sub.empty:
            continue
        ax.scatter(sub["similarity_site"], sub[param], alpha=0.6)
        # simple linear trend line
        try:
            coef = np.polyfit(sub["similarity_site"], sub[param], 1)
            xline = np.linspace(xlim[0], xlim[1], 50)
            yline = coef[0]*xline + coef[1]
            ax.plot(xline, yline, color="red", linestyle="--")
        except Exception:
            pass
        ax.set_title(f"{cond.capitalize()}")
        ax.set_xlabel("Cosine similarity")
        ax.grid(True, linestyle=":", linewidth=0.6)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    axes[0].set_ylabel(param)
    fig.suptitle(f"{param} vs Cosine similarity (Active vs Sham)")
    fig.tight_layout(rect=[0, 0.00, 1, 0.95])

    if save:
        fig.savefig(outdir / f"paired_scatter_{param}_vs_similarity.png", dpi=600)
        
    plt.close(fig)
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Usa tus defaults globales si ya los tienes (FIGSIZE_2COL, SAVE_KW, etc.)
FIGSIZE_2COL = (7.2, 4.6)
SAVE_KW = dict(dpi=600, bbox_inches="tight", pad_inches=0.01)

def correlation_matrix_active_sham_physiological(df, outdir, method="spearman", save=True):
    """
    Side-by-side correlation matrices (Active | Sham) ONLY for physiological parameters:
    ['similarity_site','A1','A2','f1','f2','gamma1','gamma2'].
    Pure matplotlib (imshow + manual annotations) so every cell shows its value.
    """
    params = [c for c in ["similarity_site","A1","A2","f1","f2","gamma1","gamma2"] if c in df.columns]
    conds = ["active", "sham"]
    available = [c for c in conds if c in df["cond"].unique()]
    if len(available) == 0 or len(params) < 2:
        return

    def _corr(subdf):
        X = subdf[params].dropna()
        if X.empty:
            return np.full((len(params), len(params)), np.nan)
        C = X.corr(method=method)
        C = C.loc[params, params]  # filas/columnas en el mismo orden
        return C.values

    mats = [ _corr(df[df["cond"]==cond]) for cond in available ]

    fig, axes = plt.subplots(1, len(available), figsize=FIGSIZE_2COL, constrained_layout=True)
    if len(available) == 1:
        axes = [axes]

    vmin, vmax = -1.0, 1.0
    for ax, cond, M in zip(axes, available, mats):
        im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap="coolwarm", interpolation="nearest")
        ax.set_xticks(range(len(params))); ax.set_yticks(range(len(params)))
        ax.set_xticklabels(params, rotation=45, ha="right"); ax.set_yticklabels(params)
        # cuadrícula
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xticks(np.arange(-.5, len(params), 1), minor=True)
        ax.set_yticks(np.arange(-.5, len(params), 1), minor=True)
        ax.grid(which="minor", color="w", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False)
        # anotar cada celda
        for i in range(len(params)):
            for j in range(len(params)):
                val = M[i, j]
                if np.isfinite(val):
                    txt_color = "white" if abs(val) > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color=txt_color, fontsize=8)
        ax.set_title(f"{cond.capitalize()}")

    cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("Spearman ρ", rotation=90)
    fig.suptitle("Spearman correlation matrices ( similarity + DDS parameters)")

    if save:
        fig.savefig(outdir / "correlation_matrix_phys_active_sham.png", **SAVE_KW)
        
    plt.close(fig)


# ----------------- main -----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root", type=str, required=True)
    ap.add_argument("--cosine-csv", type=str, required=True)
    ap.add_argument("--dds-csv", type=str,
                    help="Por defecto: <bids_root>/derivatives/dds_ds001849/dds_params_group.csv")
    ap.add_argument("--outdir", type=str,
                    help="Por defecto: <bids_root>/derivatives/compare_dds_vs_cosine")
    ap.add_argument("--save-png", action="store_true")
    ap.add_argument("--paired-boxplots", action="store_true",
                    help="Generate side-by-side (Active|Sham) boxplots with shared Y axis.")

    args = ap.parse_args()
    
    bids_root = Path(args.bids_root)
    cosine_csv = Path(args.cosine_csv)
    dds_csv = Path(args.dds_csv) if args.dds_csv else bids_root / "derivatives" / "dds_ds001849" / "dds_params_group.csv"
    outdir = Path(args.outdir) if args.outdir else bids_root / "derivatives" / "compare_dds_vs_cosine"
    outdir.mkdir(parents=True, exist_ok=True)

    # cargar datos
    cos = pd.read_csv(cosine_csv)
    dds = pd.read_csv(dds_csv)

    # colapsar cosine a nivel de sitio
    cos_site = collapse_cosine_to_site(cos)

    # agregar DDS a nivel sujeto×sitio×condición
    dds_agg = aggregate_dds(dds)

    # fusionar
    merged = pd.merge(dds_agg, cos_site, on=["subject","site","cond"], how="inner")

    # guardar merged
    merged.to_csv(outdir / "merged_dds_cosine_site.csv", index=False)

    # --------- estadísticos: OLS+FE sujeto; SE clusterizada ---------
    # 1) Modelos para similarity por sitio
    sim_df = merged[["subject","site","cond","similarity_site"]].dropna().copy()
    if not sim_df.empty:
        table_sim, model_sim = mixed_or_fe(sim_df, "similarity_site ~ C(site) + C(cond) + C(subject)")
        table_sim["family"] = "cosine_similarity"
        # FDR dentro de esta familia
        _, q_sim = fdr_bh(table_sim["p"].values)
        table_sim["p_fdr"] = q_sim
    else:
        table_sim = pd.DataFrame()

    # 2) Modelos para parámetros DDS
    params = [c for c in ["A1","gamma1","f1","A2","gamma2","f2","R2","RMSE"] if c in merged.columns]
    all_tables = []
    for p in params:
        df_long = merged[["subject","site","cond",p]].dropna().rename(columns={p:"y"})
        if df_long.empty:
            continue
        table, model = mixed_or_fe(df_long, "y ~ C(site) + C(cond) + C(subject)")
        table["param"] = p
        table["family"] = "dds_param"
        # FDR por parámetro (más estricto) o por familia (más laxo). Aquí: por familia.
        _, q = fdr_bh(table["p"].values)
        table["p_fdr"] = q
        all_tables.append(table)

    stats_tbl = pd.concat([table_sim] + all_tables, ignore_index=True) if all_tables or not table_sim.empty else pd.DataFrame()
    if not stats_tbl.empty:
        stats_tbl.to_csv(outdir / "stats_cond_site_FEcluster.csv", index=False)

    # --------- correlaciones similarity vs DDS ---------
    corr_rows = []
    for p in params:
        sub = merged[["similarity_site", p]].dropna()
        if sub.empty:
            continue
        rho, pv = spearmanr(sub["similarity_site"], sub[p])
        corr_rows.append({"param": p, "spearman_rho": rho, "p": pv})
        # Figura
        #scatter_similarity_vs_param(merged, p, outdir, save_png=args.save_png)

    corr_df = pd.DataFrame(corr_rows)
    if not corr_df.empty:
        _, q_corr = fdr_bh(corr_df["p"].values)
        corr_df["p_fdr"] = q_corr
        corr_df.to_csv(outdir / "correlations_similarity_vs_dds.csv", index=False)

# --------- paired boxplots (side-by-side Active|Sham) ---------
    if args.paired_boxplots:
        for p in [x for x in ["A1","A2","f1","f2","gamma1","gamma2","R2","RMSE"] if x in merged.columns]:
            paired_scatter_similarity_vs_param(
                merged, p, outdir, save=True
            )
        # 1) cosine similarity (si existe)
        if "similarity_site" in merged.columns:
            paired_boxplot_by_site(
                merged, "similarity_site", outdir,
                title="Cosine similarity by site (Active vs Sham)"
            )

        # 2) DDS parameters (elige el orden que prefieras en tu paper)
        for p in [x for x in ["A1","A2","f1","f2","gamma1","gamma2","R2","RMSE"] if x in merged.columns]:
            paired_boxplot_by_site(
                merged, p, outdir,
                title=f"{p} by site (Active vs Sham)"
            )
    
    # --------- correlation matrices (Active vs Sham) ---------
    correlation_matrix_active_sham_physiological(merged, outdir, method="spearman", save=True)

    # descriptivos agrupados (para tablas del paper)
    desc = merged.groupby(["site","cond"]).agg(
        n=("subject","nunique"),
        sim_mean=("similarity_site","mean"),
        sim_sd=("similarity_site","std"),
        gamma1_mean=("gamma1","mean") if "gamma1" in merged.columns else ("similarity_site","size"),
        gamma1_sd=("gamma1","std") if "gamma1" in merged.columns else ("similarity_site","size"),
        f2_mean=("f2","mean") if "f2" in merged.columns else ("similarity_site","size"),
        f2_sd=("f2","std") if "f2" in merged.columns else ("similarity_site","size"),
        R2_mean=("R2","mean") if "R2" in merged.columns else ("similarity_site","size"),
        R2_sd=("R2","std") if "R2" in merged.columns else ("similarity_site","size"),
        RMSE_mean=("RMSE","mean") if "RMSE" in merged.columns else ("similarity_site","size"),
        RMSE_sd=("RMSE","std") if "RMSE" in merged.columns else ("similarity_site","size"),
    ).reset_index()
    desc.to_csv(outdir / "descriptives_by_site_cond.csv", index=False)

    print("Hecho.")
    print("Archivos generados en:", outdir)

if __name__ == "__main__":
    main()

