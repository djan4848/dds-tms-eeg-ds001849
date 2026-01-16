import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel, wilcoxon

INP = "outputs/features/dds_plus_mi.csv"
OUT = "outputs/stats/cond_effects_by_roi.csv"

m = pd.read_csv(INP)
m = m[m["window"] == "late"].copy()

rows = []

for site, g in m.groupby("site"):
    # corr(f2, MI) por condición
    for cond in ["active", "sham"]:
        gg = g[g["cond"] == cond]
        r, p = pearsonr(gg["f2"], gg["mi_early_late_bits"])
        rows.append({
            "site": site,
            "analysis": "corr_f2_vs_mi",
            "cond": cond,
            "n": len(gg),
            "pearson_r": float(r),
            "pearson_p": float(p),
        })

    # paired: MI active vs sham (por sujeto)
    piv_mi = g.pivot_table(index="subject", columns="cond", values="mi_early_late_bits", aggfunc="mean")
    piv_mi = piv_mi.dropna(subset=["active", "sham"])
    d = piv_mi["active"] - piv_mi["sham"]

    t = ttest_rel(piv_mi["active"], piv_mi["sham"])
    rows.append({
        "site": site,
        "analysis": "paired_MI_active_minus_sham",
        "cond": "active-sham",
        "n": len(piv_mi),
        "mean_diff": float(d.mean()),
        "median_diff": float(d.median()),
        "t": float(t.statistic),
        "p": float(t.pvalue),
    })
    try:
        w = wilcoxon(piv_mi["active"], piv_mi["sham"])
        rows.append({
            "site": site,
            "analysis": "wilcoxon_MI_active_minus_sham",
            "cond": "active-sham",
            "n": len(piv_mi),
            "W": float(w.statistic),
            "p": float(w.pvalue),
        })
    except Exception as e:
        rows.append({"site": site, "analysis": "wilcoxon_MI_active_minus_sham", "cond": "active-sham", "error": repr(e)})

    # paired: f2 active vs sham
    piv_f2 = g.pivot_table(index="subject", columns="cond", values="f2", aggfunc="mean")
    piv_f2 = piv_f2.dropna(subset=["active", "sham"])
    d2 = piv_f2["active"] - piv_f2["sham"]
    t2 = ttest_rel(piv_f2["active"], piv_f2["sham"])
    rows.append({
        "site": site,
        "analysis": "paired_f2_active_minus_sham",
        "cond": "active-sham",
        "n": len(piv_f2),
        "mean_diff": float(d2.mean()),
        "median_diff": float(d2.median()),
        "t": float(t2.statistic),
        "p": float(t2.pvalue),
    })
    try:
        w2 = wilcoxon(piv_f2["active"], piv_f2["sham"])
        rows.append({
            "site": site,
            "analysis": "wilcoxon_f2_active_minus_sham",
            "cond": "active-sham",
            "n": len(piv_f2),
            "W": float(w2.statistic),
            "p": float(w2.pvalue),
        })
    except Exception as e:
        rows.append({"site": site, "analysis": "wilcoxon_f2_active_minus_sham", "cond": "active-sham", "error": repr(e)})

out = pd.DataFrame(rows)
out.to_csv(OUT, index=False)
print("[OK] wrote:", OUT)
print(out)
