import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_rel, wilcoxon

INP = "outputs/features/dds_plus_mi.csv"

def fisher_z(r):
    r = np.clip(r, -0.999999, 0.999999)
    return 0.5 * np.log((1+r)/(1-r))

m = pd.read_csv(INP)

# usamos window late para f2 y MI (MI es única por subj/site/cond)
m = m[m["window"] == "late"].copy()

# centramos en M1 (donde viste efecto)
m1 = m[m["site"] == "m1"].copy()

# --- A) correlación f2 vs MI por condición ---
def corr_block(df, label):
    r, p = pearsonr(df["f2"], df["mi_early_late_bits"])
    print(f"{label}: r={r:+.3f} p={p:.4g} n={len(df)}")
    return r, p

act = m1[m1["cond"] == "active"]
shm = m1[m1["cond"] == "sham"]

print("\n=== M1: corr(f2, MI) por condición ===")
rA, pA = corr_block(act, "ACTIVE")
rS, pS = corr_block(shm, "SHAM")

# Comparación aproximada de correlaciones (Fisher z, asumiendo independencia; útil como quick check)
# Nota: ACTIVE y SHAM comparten sujetos -> la comparación exacta requiere métodos para correlaciones dependientes.
zA, zS = fisher_z(rA), fisher_z(rS)
nA, nS = len(act), len(shm)
se = np.sqrt(1/(nA-3) + 1/(nS-3))
z = (zA - zS) / se
print(f"Fisher z (aprox, indep): z={z:+.3f} (ojo: approx)")

# --- B) paired: MI_active vs MI_sham por sujeto ---
print("\n=== M1: paired MI (ACTIVE - SHAM) ===")
piv = m1.pivot_table(index="subject", columns="cond", values="mi_early_late_bits", aggfunc="mean")
piv = piv.dropna(subset=["active","sham"])
d = piv["active"] - piv["sham"]
print(f"n_subjects={len(piv)} mean_diff={d.mean():+.4f} median_diff={d.median():+.4f}")

t = ttest_rel(piv["active"], piv["sham"])
print(f"paired t-test: t={t.statistic:+.3f} p={t.pvalue:.4g}")

try:
    w = wilcoxon(piv["active"], piv["sham"])
    print(f"wilcoxon: W={w.statistic:.3f} p={w.pvalue:.4g}")
except Exception as e:
    print("wilcoxon failed:", repr(e))

# --- C) paired: f2_active vs f2_sham por sujeto ---
print("\n=== M1: paired f2 (ACTIVE - SHAM) ===")
piv2 = m1.pivot_table(index="subject", columns="cond", values="f2", aggfunc="mean")
piv2 = piv2.dropna(subset=["active","sham"])
d2 = piv2["active"] - piv2["sham"]
print(f"n_subjects={len(piv2)} mean_diff={d2.mean():+.4f} median_diff={d2.median():+.4f}")

t2 = ttest_rel(piv2["active"], piv2["sham"])
print(f"paired t-test: t={t2.statistic:+.3f} p={t2.pvalue:.4g}")

try:
    w2 = wilcoxon(piv2["active"], piv2["sham"])
    print(f"wilcoxon: W={w2.statistic:.3f} p={w2.pvalue:.4g}")
except Exception as e:
    print("wilcoxon failed:", repr(e))
