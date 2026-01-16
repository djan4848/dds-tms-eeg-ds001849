import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

INP = "outputs/features/info_mutual_information_channelwise.csv"
OUT = "outputs/stats/mi_channelwise_active_minus_sham.csv"

def bh_fdr(pvals):
    pvals = np.asarray(pvals, float)
    n = np.sum(np.isfinite(pvals))
    out = np.full_like(pvals, np.nan, dtype=float)
    if n == 0: return out
    idx = np.where(np.isfinite(pvals))[0]
    pv = pvals[idx]
    order = np.argsort(pv)
    pv_sorted = pv[order]
    q = pv_sorted * n / (np.arange(1, n+1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    out_idx = np.empty_like(q)
    out_idx[order] = q
    out[idx] = np.clip(out_idx, 0, 1)
    return out

df = pd.read_csv(INP)
df = df[df["reason"] == "ok"].copy()

rows = []
for (site, ch), g in df.groupby(["site","channel"]):
    piv = g.pivot_table(index="subject", columns="cond", values="mi_early_late_bits", aggfunc="mean")
    if not {"active","sham"} <= set(piv.columns):
        continue
    piv = piv.dropna(subset=["active","sham"])
    if len(piv) < 8:
        continue

    d = piv["active"] - piv["sham"]

    t = ttest_rel(piv["active"], piv["sham"])
    try:
        w = wilcoxon(piv["active"], piv["sham"])
        wp = w.pvalue
        W = float(w.statistic)
    except Exception:
        wp = np.nan
        W = np.nan

    rows.append({
        "site": site,
        "channel": ch,
        "n_subjects": int(len(piv)),
        "mean_diff_active_minus_sham": float(d.mean()),
        "median_diff_active_minus_sham": float(d.median()),
        "t": float(t.statistic),
        "p_t": float(t.pvalue),
        "W": W,
        "p_wilcoxon": float(wp) if np.isfinite(wp) else np.nan,
    })

out = pd.DataFrame(rows)

# FDR por site (sobre p_t)
out["q_t_site"] = np.nan
for site, idx in out.groupby("site").groups.items():
    out.loc[idx, "q_t_site"] = bh_fdr(out.loc[idx, "p_t"].values)

# FDR global
out["q_t_global"] = bh_fdr(out["p_t"].values)

out.to_csv(OUT, index=False)
print("[OK] wrote:", OUT)
print("Top hits (q_t_site <= 0.05):")
hits = out[out["q_t_site"] <= 0.05].sort_values(["site","q_t_site","p_t"])
print(hits.head(50).to_string(index=False))
