#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ttest_ind
# Importa tu DDS (para reconstruir y sensibilidad)
from scripts.dds_model import fit_dds  # params + y_hat


# -------------------------
# Utils
# -------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p

def load_csvs(deriv_root: Path):
    df_ch  = pd.read_csv(deriv_root / "dds_params_channelwise.csv")
    df_roi = pd.read_csv(deriv_root / "dds_params_roi.csv")
    return df_ch, df_roi

def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def mean_ci_t(x, alpha=0.05):
    """
    Mean ± t-based 95% CI.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        m = float(np.nanmean(x)) if n == 1 else np.nan
        return m, np.nan, np.nan, n
    m = np.mean(x)
    s = np.std(x, ddof=1) / np.sqrt(n)
    from scipy.stats import t
    h = s * t.ppf(1 - alpha/2, n-1)
    return float(m), float(m - h), float(m + h), int(n)

def bootstrap_ci_mean(x, n_boot=5000, alpha=0.05, seed=0):
    """
    Bootstrap CI for the mean.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        samp = rng.choice(x, size=len(x), replace=True)
        means[i] = np.mean(samp)
    lo = np.percentile(means, 100*alpha/2)
    hi = np.percentile(means, 100*(1-alpha/2))
    return float(np.mean(x)), float(lo), float(hi), int(len(x))

def cohens_d_independent(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled = np.sqrt(((nx-1)*vx + (ny-1)*vy) / (nx+ny-2))
    if pooled == 0:
        return np.nan
    return float((np.mean(x) - np.mean(y)) / pooled)


# -------------------------
# Fig 0: Fit quality
# -------------------------
def fig0_r2_hist(df_ch: pd.DataFrame, outdir: Path):
    plt.figure()
    plt.hist(df_ch["R2"].dropna().values, bins=60)
    plt.xlabel("R² (channel-wise)")
    plt.ylabel("count")
    plt.title("DDS fit quality (channel-wise)")
    savefig(outdir / "Fig0_R2_hist_channelwise.png")


# -------------------------
# Fig 1: Parameter sensitivity (didactic)
# -------------------------
def fig1_parameter_sensitivity(example_sici_fif: Path, example_channel: str, outdir: Path):
    """
    Didactic sensitivity plot (not inferential).
    """
    import mne

    epochs = mne.read_epochs(str(example_sici_fif), preload=True, verbose="ERROR")
    epochs.apply_baseline((-0.5, 0.0))
    evk = epochs.average()
    ch_ix = evk.ch_names.index(example_channel)
    t = evk.times
    y = evk.data[ch_ix, :]

    params, yhat = fit_dds(t, y, tmin=0.010, tmax=0.100, two_components=True, t0=0.0, h_freq=100.0)

    keys = ["A1","gamma1","f1","phi1","A2","gamma2","f2","phi2","offset"]
    base = {k: params[k] for k in keys}

    fig, axes = plt.subplots(3, 3, figsize=(10, 7), sharex=True, sharey=True)
    axes = axes.ravel()

    def plot_scaled(ax, key):
        ax.plot(t*1000, y, linewidth=1, label="TEP" if key == "A1" else None)
        ax.plot(t*1000, yhat, linewidth=1, label="DDS fit" if key == "A1" else None)

        if key in ["A1","A2","offset"]:
            ax.plot(t*1000, yhat * 1.2, linewidth=1)
            ax.plot(t*1000, yhat * 0.8, linewidth=1)
        elif key in ["gamma1","gamma2"]:
            tt = np.clip(t, 0, None)
            env = np.exp(-base[key]*tt)
            ax.plot(t*1000, (yhat - base["offset"]) * env + base["offset"], linewidth=1)
            ax.plot(t*1000, (yhat - base["offset"]) * np.exp(-(base[key]*0.8)*tt) + base["offset"], linewidth=1)
        elif key in ["f1","f2"]:
            tt = np.linspace(t.min(), t.max(), len(t))
            y_interp = np.interp(tt, t, yhat)
            y_fast = np.interp(tt, t*0.9, y_interp, left=y_interp[0], right=y_interp[-1])
            y_slow = np.interp(tt, t*1.1, y_interp, left=y_interp[0], right=y_interp[-1])
            ax.plot(tt*1000, y_fast, linewidth=1)
            ax.plot(tt*1000, y_slow, linewidth=1)

        ax.set_title(key)
        ax.axvline(0, linewidth=0.8)
        ax.set_xlim(0, 120)

    for i, key in enumerate(["A1","A2","gamma1","gamma2","f1","f2","phi1","phi2","offset"]):
        plot_scaled(axes[i], key)

    for ax in axes[6:]:
        ax.set_xlabel("Time (ms)")
    for ax in axes[::3]:
        ax.set_ylabel("Amplitude (a.u.)")

    axes[0].legend(loc="best")
    plt.suptitle("DDS parameter sensitivity (didactic)")
    plt.tight_layout(rect=[0,0,1,0.96])
    plt.savefig(outdir / "Fig1_parameter_sensitivity_didactic.png", dpi=300)
    plt.close()


# -------------------------
# Fig 2: Example fit (SICI + ICF, same subject/channel)
# -------------------------
def fig2_example_fit_pair(example_sici_fif: Path, example_icf_fif: Path, example_channel: str, outdir: Path):
    """
    Example fit with explicit legend:
    TEP = blue, DDS fit = orange.
    """
    import mne

    def load_fit(fif, tmin, tmax):
        epochs = mne.read_epochs(str(fif), preload=True, verbose="ERROR")
        epochs.apply_baseline((-0.5, 0.0))
        evk = epochs.average()
        ch_ix = evk.ch_names.index(example_channel)
        t = evk.times
        y = evk.data[ch_ix, :]
        params, yhat = fit_dds(t, y, tmin=tmin, tmax=tmax, two_components=True, t0=0.0, h_freq=100.0)
        return t, y, yhat, params

    t_s, y_s, yhat_s, ps = load_fit(example_sici_fif, 0.010, 0.100)
    t_i, y_i, yhat_i, pi = load_fit(example_icf_fif,  0.010, 0.200)

    plt.figure(figsize=(11, 4))

    plt.subplot(1,2,1)
    plt.plot(t_s*1000, y_s, linewidth=1, color="tab:blue", label="TEP")
    plt.plot(t_s*1000, yhat_s, linewidth=1, color="tab:orange", label="DDS fit")
    plt.axvline(0, linewidth=0.8)
    plt.title(f"SICI fit ({example_channel})\nR²={ps['R2']:.3f}, RMSE={ps['RMSE']:.3g}")
    plt.xlabel("Time (ms)"); plt.ylabel("Amplitude")
    plt.legend(loc="best")

    plt.subplot(1,2,2)
    plt.plot(t_i*1000, y_i, linewidth=1, color="tab:blue", label="TEP")
    plt.plot(t_i*1000, yhat_i, linewidth=1, color="tab:orange", label="DDS fit")
    plt.axvline(0, linewidth=0.8)
    plt.title(f"ICF fit ({example_channel})\nR²={pi['R2']:.3f}, RMSE={pi['RMSE']:.3g}")
    plt.xlabel("Time (ms)"); plt.ylabel("Amplitude")
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(outdir / "Fig2_example_fit_SICI_and_ICF.png", dpi=300)
    plt.close()


# -------------------------
# Fig 3: Protocol signatures (pooled, descriptive)
# -------------------------
def fig3_protocol_signatures_pooled(df_ch: pd.DataFrame, outdir: Path):
    """
    Descriptive protocol signatures:
      - SICI: gamma1 pooled over all channels/subjects
      - ICF:  f2 pooled over all channels/subjects
    Adds bootstrap 95% CI for the mean.
    """
    d = df_ch[df_ch["group"].isin(["CTL","OCD"])].copy()

    sici_g1 = d[(d.protocol=="SICI")]["gamma1"].dropna().values
    icf_f2  = d[(d.protocol=="ICF")]["f2"].dropna().values

    plt.figure(figsize=(11,4))

    plt.subplot(1,2,1)
    plt.violinplot(sici_g1, showmeans=False, showextrema=False)
    m, lo, hi, n = bootstrap_ci_mean(sici_g1, n_boot=5000, seed=0)
    plt.errorbar(1.0, m, yerr=[[m-lo],[hi-m]], fmt='o', color='black', capsize=6, linewidth=2,
                 label="Mean ± 95% bootstrap CI")
    plt.title(f"SICI signature: pooled γ₁ (all channels)\nN={n} channel-samples")
    plt.ylabel("γ₁ (s⁻¹)")
    plt.xticks([1], ["SICI"])
    plt.legend(loc="best")

    plt.subplot(1,2,2)
    plt.violinplot(icf_f2, showmeans=False, showextrema=False)
    m, lo, hi, n = bootstrap_ci_mean(icf_f2, n_boot=5000, seed=1)
    plt.errorbar(1.0, m, yerr=[[m-lo],[hi-m]], fmt='o', color='black', capsize=6, linewidth=2,
                 label="Mean ± 95% bootstrap CI")
    plt.title(f"ICF signature: pooled f₂ (all channels)\nN={n} channel-samples")
    plt.ylabel("f₂ (Hz)")
    plt.xticks([1], ["ICF"])
    plt.legend(loc="best")

    plt.tight_layout()
    plt.savefig(outdir / "Fig3_protocol_signatures_pooled_gamma1_SICI_f2_ICF.png", dpi=300)
    plt.close()


# -------------------------
# Fig 4: Between-group ROI (SICI gamma1)
# -------------------------
def fig4_between_roi_sici_gamma1(df_roi: pd.DataFrame, outdir: Path, rois: list[str]):
    """
    Main negative-result figure (SICI):
      CTL vs OCD in gamma1, across selected ROIs.
    Shows mean ± 95% CI + Cohen's d + Welch p (uncorrected; stats in CSV handle FDR).
    """
    d = df_roi[(df_roi["protocol"]=="SICI") & (df_roi["group"].isin(["CTL","OCD"]))].copy()

    plt.figure(figsize=(12, 4))
    for i, roi in enumerate(rois, start=1):
        sub = d[d["roi"]==roi]
        x = sub[sub["group"]=="CTL"]["gamma1"].values
        y = sub[sub["group"]=="OCD"]["gamma1"].values
        t, p = ttest_ind(x, y, equal_var=False, nan_policy="omit")
        d_eff = cohens_d_independent(x, y)

        m_x, lo_x, hi_x, nx = mean_ci_t(x)
        m_y, lo_y, hi_y, ny = mean_ci_t(y)

        plt.subplot(1, len(rois), i)
        # group points
        plt.scatter(np.zeros_like(x), x, s=18, alpha=0.7, label="CTL" if i == 1 else None)
        plt.scatter(np.ones_like(y),  y, s=18, alpha=0.7, label="OCD" if i == 1 else None)

        # CI bars
        plt.errorbar([0,1], [m_x, m_y],
                     yerr=[[m_x-lo_x, m_y-lo_y], [hi_x-m_x, hi_y-m_y]],
                     fmt='o', color='black', capsize=6, linewidth=2, label="Mean ± 95% CI" if i == 1 else None)

        plt.xticks([0,1], ["CTL","OCD"])
        plt.title(f"{roi}\nWelch p={p:.3g}, d={d_eff:.2f}\nN={nx}/{ny}")
        plt.ylabel("γ₁ (s⁻¹)")

    plt.suptitle("SICI: between-group (CTL vs OCD) in γ₁ across ROIs")
    plt.legend(loc="best")
    plt.tight_layout(rect=[0,0,1,0.92])
    plt.savefig(outdir / "Fig4_between_ROI_SICI_gamma1_CTL_vs_OCD.png", dpi=300)
    plt.close()


# -------------------------
# Fig 5: Between-group ROI (ICF f2)
# -------------------------
def fig5_between_roi_icf_f2(df_roi: pd.DataFrame, outdir: Path, rois: list[str]):
    """
    Main negative-result figure (ICF):
      CTL vs OCD in f2, across selected ROIs.
    Shows mean ± 95% CI + Cohen's d + Welch p (uncorrected).
    """
    d = df_roi[(df_roi["protocol"]=="ICF") & (df_roi["group"].isin(["CTL","OCD"]))].copy()

    plt.figure(figsize=(12, 4))
    for i, roi in enumerate(rois, start=1):
        sub = d[d["roi"]==roi]
        x = sub[sub["group"]=="CTL"]["f2"].values
        y = sub[sub["group"]=="OCD"]["f2"].values
        t, p = ttest_ind(x, y, equal_var=False, nan_policy="omit")
        d_eff = cohens_d_independent(x, y)

        m_x, lo_x, hi_x, nx = mean_ci_t(x)
        m_y, lo_y, hi_y, ny = mean_ci_t(y)

        plt.subplot(1, len(rois), i)
        plt.scatter(np.zeros_like(x), x, s=18, alpha=0.7, label="CTL" if i == 1 else None)
        plt.scatter(np.ones_like(y),  y, s=18, alpha=0.7, label="OCD" if i == 1 else None)

        plt.errorbar([0,1], [m_x, m_y],
                     yerr=[[m_x-lo_x, m_y-lo_y], [hi_x-m_x, hi_y-m_y]],
                     fmt='o', color='black', capsize=6, linewidth=2, label="Mean ± 95% CI" if i == 1 else None)

        plt.xticks([0,1], ["CTL","OCD"])
        plt.title(f"{roi}\nWelch p={p:.3g}, d={d_eff:.2f}\nN={nx}/{ny}")
        plt.ylabel("f₂ (Hz)")

    plt.suptitle("ICF: between-group (CTL vs OCD) in f₂ across ROIs")
    plt.legend(loc="best")
    plt.tight_layout(rect=[0,0,1,0.92])
    plt.savefig(outdir / "Fig5_between_ROI_ICF_f2_CTL_vs_OCD.png", dpi=300)
    plt.close()


# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deriv_root", default="derivatives/dds_gabacog", type=str)
    ap.add_argument("--example_sici_fif", default="", type=str)
    ap.add_argument("--example_icf_fif", default="", type=str)
    ap.add_argument("--example_channel", default="C3", type=str)
    ap.add_argument("--rois", default="DLPFC_L,DLPFC_R,FRONTAL_MID,PARIETAL_MID", type=str,
                    help="Comma-separated ROI names to plot in between-group figures.")
    args = ap.parse_args()

    deriv_root = Path(args.deriv_root)
    outdir = ensure_dir(deriv_root / "figures_paper_A")

    df_ch, df_roi = load_csvs(deriv_root)

    # Fig0: fit quality
    fig0_r2_hist(df_ch, outdir)

    # Fig1/Fig2 require example FIFs
        # ---- Resolve example FIFs
    # If user does not provide paths, use config DIRs to auto-pick an example.
    example_sici = args.example_sici_fif.strip() if args.example_sici_fif else ""
    example_icf  = args.example_icf_fif.strip()  if args.example_icf_fif  else ""

    if not (example_sici and example_icf):
        try:
            from configs.gabacog_sici_icf import DIR_SICI, DIR_ICF
            # Prefer sub_005 if present
            cand_sici = Path(DIR_SICI) / "sub_005__SICI-epo.fif"
            cand_icf  = Path(DIR_ICF)  / "sub_005__ICF-epo.fif"

            if cand_sici.exists():
                example_sici = str(cand_sici)
            else:
                fs = sorted(Path(DIR_SICI).glob("sub_*__SICI-epo.fif"))
                example_sici = str(fs[0]) if fs else ""

            if cand_icf.exists():
                example_icf = str(cand_icf)
            else:
                fi = sorted(Path(DIR_ICF).glob("sub_*__ICF-epo.fif"))
                example_icf = str(fi[0]) if fi else ""

            if example_sici and example_icf:
                print(f"[FIG] Auto example SICI: {example_sici}")
                print(f"[FIG] Auto example ICF : {example_icf}")
        except Exception as e:
            print(f"[FIG] Could not auto-resolve example FIFs from config: {repr(e)}")

    # Fig1/Fig2
    if example_sici and example_icf and Path(example_sici).exists() and Path(example_icf).exists():
        fig1_parameter_sensitivity(Path(example_sici), args.example_channel, outdir)
        fig2_example_fit_pair(Path(example_sici), Path(example_icf), args.example_channel, outdir)
    else:
        print("Skipping Fig1/Fig2 (example FIFs not provided or not found).")


    # Fig3: protocol signatures (descriptive)
    fig3_protocol_signatures_pooled(df_ch, outdir)

    # Fig4/5: between-group ROI (negative results reported honestly)
    rois = [r.strip() for r in args.rois.split(",") if r.strip()]
    fig4_between_roi_sici_gamma1(df_roi, outdir, rois)
    fig5_between_roi_icf_f2(df_roi, outdir, rois)

    print("Saved figures to:", outdir)


if __name__ == "__main__":
    main()

