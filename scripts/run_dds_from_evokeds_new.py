#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDS fitter for TMS-EEG Evokeds with SITE-ROI aggregation (fallback to EEG mean).
Enhanced with TEP-optimized early window fitting.

Addresses:
- EEG-only picking (no MEG/EOG/etc.)
- Evoked selection by comment matching (site/cond) to avoid ambiguity
- Units: Volts -> microvolts for outputs/plots
- Spatial aggregation by ROI per site (NOT all channels). If ROI channels not found -> fallback to mean of all EEG.
- TEP-optimized fitting for early window (15-80ms) with multi-criteria scoring and phase alignment

Outputs:
- Per-window CSV with DDS params per file
- Optional figures
- Group CSV accumulating all windows
"""

import argparse
import sys
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import mne
from scipy.optimize import curve_fit
from collections import defaultdict
from pathlib import Path

# -------------------- ROI config --------------------
ROI_DEFAULT = ["C3", "FC1", "CP1", "FC5","CP5"]
ROI_MAP = {
    "m1":    ["C3", "FC1", "CP1", "CP5","FC5"],  # left M1
    "dlpfc": ["F4", "Fp2", "F8", "FC2", "FC6"],  # right dlpfc
    "ppc":   ["P3", "P7", "CP1", "CP5", "Pz"]    # left ppc
}

# -------------------- Window Configuration --------------------
WINDOW_CONFIG = {
    "early": {"tmin": 0.015, "tmax": 0.080, "label": "15-80ms"},
    "late": {"tmin": 0.080, "tmax": 0.200, "label": "80-200ms"}
}

# -------------------- DDS model --------------------
def dds_func(t, A1, gamma1, f1, A2, gamma2, f2):
    return (
        A1 * np.exp(-gamma1 * t) * np.sin(2 * np.pi * f1 * t) +
        A2 * np.exp(-gamma2 * t) * np.sin(2 * np.pi * f2 * t)
    )

# ===================== JOINT GRAND-AVERAGE FIGURE =====================
def _percentile_ylim(arrs, low=2, high=98, pad=0.05):
    import numpy as np
    vals = np.concatenate([a.ravel() for a in arrs if a.size > 0])
    if vals.size == 0:
        return (-1, 1)
    lo, hi = np.percentile(vals, [low, high])
    span = hi - lo if hi > lo else 1.0
    return (lo - pad*span, hi + pad*span)

def plot_joint_grand_average(ga_traces, ga_times, window_label, out_path,
                             sites=("m1", "dlpfc", "ppc"),
                             conds=("active", "sham"),
                             colors={"active":"#cc6e6e", "sham":"#8aa6c1"}):
    """
    Make a 2x3 panel: rows=sites (M1,DLPFC,PPC), cols=conditions (Active,Sham).
    Each panel: TEP mean ±1SD + DDS fit on the mean (same timebase).
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Collect all traces to compute a global y-limit per window
    all_means = []
    for s in sites:
        for c in conds:
            key = (s, c, window_label)
            if key in ga_traces and len(ga_traces[key]) > 0:
                Y = np.vstack(ga_traces[key])            # n_subj x n_time
                all_means.append(Y.mean(axis=0))
    
    if not all_means:
        print(f"[WARN] No data for joint grand average: {window_label}")
        return
        
    ylo, yhi = _percentile_ylim(all_means, low=2, high=98, pad=0.08)

    fig, axes = plt.subplots(len(sites), len(conds),
                             figsize=(12.5, 6.8),
                             sharex=True, sharey=True)
    if axes.ndim == 1:
        axes = axes.reshape(len(sites), len(conds))

    for i, s in enumerate(sites):
        for j, c in enumerate(conds):
            ax = axes[i, j]
            key = (s, c, window_label)
            if key not in ga_traces or len(ga_traces[key]) == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", fontsize=10)
                ax.set_axis_off()
                continue

            t = ga_times[key]
            tt_ms = t * 1e3
            Y = np.vstack(ga_traces[key])                # n_subj x n_time
            mean_uv = Y.mean(axis=0)
            sd_uv   = Y.std(axis=0, ddof=1)

            # DDS fit on the grand mean
            try:
                w0, w1 = window_label.replace("ms","").split("-")
                tmin_s, tmax_s = float(w0)/1000.0, float(w1)/1000.0
            except Exception:
                tmin_s, tmax_s = float(t[0]), float(t[-1])

            popt, r2, rmse_uv = fit_dds(t, mean_uv, tmin=tmin_s, tmax=tmax_s)
            yhat = dds_func(t, *popt)

            # Plot mean ± SD and DDS fit
            ax.plot(tt_ms, mean_uv, lw=2, color=colors[c], label=f"{c.capitalize()} mean")
            ax.fill_between(tt_ms, mean_uv - sd_uv, mean_uv + sd_uv,
                            color=colors[c], alpha=0.22, linewidth=0, label="±1 SD")
            ax.plot(tt_ms, yhat, "--", lw=2, color="black", label="DDS fit")

            # cosmetics
            ax.set_ylim(ylo, yhi)
            ax.grid(alpha=0.25)
            if i == len(sites)-1:
                ax.set_xlabel("Time (ms)")
            if j == 0:
                ax.set_ylabel("Amplitude (µV)")
            site_title = s.upper()
            ax.set_title(f"{site_title} • {c.capitalize()}  (R²={r2:.2f}, RMSE={rmse_uv:.1f} µV)", fontsize=10)

    # A single legend
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Grand-average TEPs with DDS fit • Window {window_label}", y=0.995, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Joint grand-average figure saved: {out_path}")
# =====================================================================

# -------------------- Enhanced TEP-Optimized Fitting --------------------
def _clip_p0(p0, bounds, eps=1e-6):
    lb, ub = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    p0 = np.asarray(p0, float)
    return np.minimum(ub - eps, np.maximum(lb + eps, p0)).tolist()

def _rmse(y, yhat):
    r = y - yhat
    return float(np.sqrt(np.mean(r**2)))

def _fit_once(t, y, p0, bounds, maxfev):
    p0 = _clip_p0(p0, bounds)
    popt, _ = curve_fit(dds_func, t, y, p0=p0, bounds=bounds, maxfev=maxfev)
    yhat = dds_func(t, *popt)
    return popt, _rmse(y, yhat)

def _early_tep_config(peak, tmin, tmax):
    """
    Optimized for early TEP (15-80ms) characteristics with physiological constraints
    """
    # More constrained bounds for better R2 and physiological plausibility
    a1_lo, a1_hi = max(0.1, 0.3 * peak), min(1000.0, 2.0 * peak)  # Reduced upper bound
    a2_lo, a2_hi = -min(800.0, 1.2 * peak), min(800.0, 1.2 * peak)  # More constrained
    
    return {
        "bounds": (
            [a1_lo, 50.0, 8.0,   a2_lo, 20.0, 45.0],   # Tighter lower bounds
            [a1_hi, 300.0, 25.0,  a2_hi, 100.0, 120.0]  # Tighter upper bounds
        ),
        "gam1_grid": [80.0, 120.0, 180.0, 240.0],  # More focused range
        "gam2_grid": [25.0, 40.0, 60.0, 80.0],     
        "f1_grid": [4.0, 12.0, 16.0, 30.0],        # Theta-beta range prev 8.0, 12.0,16.0,20.0
        "f2_grid": [40.0, 80.0, 100.0],            # Gamma range
        "a1_frac": [0.5, 0.8, 1.0, 1.2],          # More conservative amplitudes
        "a2_frac": [-0.2, -0.1, 0.1, 0.2]         # Smaller relative amplitudes
    }

def _early_tep_candidates(peak, config):
    """Generate candidates specifically for TEP early components"""
    cands = []
    
    for a1_frac in config["a1_frac"]:
        for a2_frac in config["a2_frac"]:
            for g1 in config["gam1_grid"]:
                for g2 in config["gam2_grid"]:
                    for f1 in config["f1_grid"]:
                        for f2 in config["f2_grid"]:
                            cands.append([
                                a1_frac * peak,  # A1
                                g1,              # gamma1  
                                f1,              # f1
                                a2_frac * peak,  # A2
                                g2,              # gamma2
                                f2               # f2
                            ])
    return cands

def _tep_phase_aware_fit(t, y, candidates, bounds, maxfev):
    """
    TEP-optimized fitting that considers phase relationships and R2 optimization
    """
    early_mask = (t >= 0.015) & (t <= 0.035)  # N1/P1 window
    late_early_mask = (t >= 0.040) & (t <= 0.080)  # N2/P2 window
    full_mask = (t >= 0.015) & (t <= 0.080)   # Full early TEP
    
    best_params = None
    best_score = -np.inf  # Now maximizing R2-based score
    best_r2 = -np.inf
    
    for p0 in candidates:
        try:
            popt, rmse_full = _fit_once(t, y, p0, bounds, maxfev)
            yhat = dds_func(t, *popt)
            
            # Calculate R2 for different segments
            ss_res_full = np.sum((y[full_mask] - yhat[full_mask])**2)
            ss_tot_full = np.sum((y[full_mask] - np.mean(y[full_mask]))**2) + 1e-12
            r2_full = 1.0 - ss_res_full/ss_tot_full
            
            # Multi-criteria scoring prioritizing R2
            rmse_early = _rmse(y[early_mask], yhat[early_mask]) if early_mask.any() else rmse_full
            rmse_late = _rmse(y[late_early_mask], yhat[late_early_mask]) if late_early_mask.any() else rmse_full
            
            # Phase alignment score
            cc_early = np.corrcoef(y[early_mask], yhat[early_mask])[0,1] if early_mask.any() and len(y[early_mask]) > 1 else 0
            cc_late = np.corrcoef(y[late_early_mask], yhat[late_early_mask])[0,1] if late_early_mask.any() and len(y[late_early_mask]) > 1 else 0
            
            # Combined score prioritizing R2 and physiological plausibility
            score = (0.5 * r2_full +                    # Main R2 component
                    0.2 * (1 - rmse_early/rmse_full) +  # Early component fit
                    0.2 * (1 - rmse_late/rmse_full) +   # Late component fit  
                    0.05 * cc_early +                   # Phase alignment
                    0.05 * cc_late)                     # Phase alignment
            
            # Strong preference for high R2 values
            if r2_full > 0.7:
                score += 0.3
            elif r2_full > 0.5:
                score += 0.1
                
            if score > best_score and np.isfinite(score) and r2_full > best_r2:
                best_params, best_score, best_r2 = popt, score, r2_full
                
        except Exception:
            continue
            
    return best_params

def _refine_tep_fit(t, y, initial_params, bounds, maxfev):
    """Refine TEP fit focusing on R2 improvement"""
    best_params = initial_params
    yhat_initial = dds_func(t, *initial_params)
    best_rmse = _rmse(y, yhat_initial)
    
    # More conservative refinement for better convergence
    refinement_scales = np.array([0.05, 0.03, 0.03, 0.08, 0.05, 0.05])  # Smaller perturbations
    
    for _ in range(15):  # Fewer refinements
        try:
            perturbation = (np.random.rand(6) - 0.5) * 2.0 * refinement_scales
            candidate = best_params * (1.0 + perturbation)
            candidate = _clip_p0(candidate, bounds)
            
            popt, rmse = _fit_once(t, y, candidate, bounds, maxfev=3000)
            
            if rmse < best_rmse:
                best_params, best_rmse = popt, rmse
                
        except Exception:
            continue
            
    return best_params

def _validate_tep_fit(params, y, yhat, t):
    """Enhanced validation with stricter physiological constraints"""
    A1, g1, f1, A2, g2, f2 = params
    
    # Stricter physiological plausibility checks
    checks = [
        (abs(A1) < 500, "A1 amplitude reasonable (<500 µV)"),           
        (abs(A2) < 500, "A2 amplitude reasonable (<500 µV)"),
        (4 <= f1 <= 30, "f1 in theta- high beta range (8-30 Hz)"),       
        (45 <= f2 <= 120, "f2 in gamma range (45-120 Hz)"),
        (g1 > 0 and g2 > 0, "positive decay constants"),
        (g1 < 500 and g2 < 500, "reasonable decay rates (<500)"),
        (abs(A1) > 0.1, "A1 non-trivial"),
        (abs(A2) > 0.1, "A2 non-trivial")
    ]
    
    # Fit quality check with higher threshold
    r_squared = 1 - np.sum((y - yhat)**2) / (np.sum((y - np.mean(y))**2) + 1e-12)
    checks.append((r_squared > -0.5, "Reasonable fit quality (R² > -0.5)"))
    
    failed_checks = [check[1] for check in checks if not check[0]]
    return len(failed_checks) == 0, failed_checks

def _late_config(peak):
    """Configuration for late window (80-200ms)"""
    return {
        "bounds": ([-800, 1.0, 4.0,  -800, 1.0, 45.0],   # Tighter bounds
                   [ 800, 15.0, 12.0,  800, 15.0, 80.0]), # Reduced upper bounds
        "p0": [0.40*peak, 3.0, 8.0, 0.40*peak, 3.0, 55.0], # More conservative initial
        "gam1_grid": [2.0, 4.0, 6.0, 8.0],
        "gam2_grid": [2.0, 4.0, 6.0, 8.0],
        "f1_grid":   [6.0, 8.0, 10.0],
        "f2_grid":   [50.0, 55.0, 60.0, 65.0],
    }

def fit_dds(time_s, y, tmin, tmax, p0=None, bounds=None, maxfev=150000):
    """
    Enhanced DDS fit for TEP data with R2 optimization and physiological constraints
    """
    t = np.asarray(time_s, float)
    y = np.asarray(y, float)
    peak = float(np.nanmax(np.abs(y))) if np.isfinite(y).any() else 1.0

    # EARLY TEP WINDOW (15-80ms)
    if tmax <= 0.080:  # Changed from 0.80 to 0.080 for 80ms
        tep_config = _early_tep_config(peak, tmin, tmax)
        if bounds is None:
            bounds = tep_config["bounds"]
        
        candidates = _early_tep_candidates(peak, tep_config)
        if p0 is not None:
            candidates.insert(0, p0)

        best = _tep_phase_aware_fit(t, y, candidates, bounds, maxfev)
        
        if best is not None:
            refined = _refine_tep_fit(t, y, best, bounds, maxfev)
            if refined is not None:
                best = refined

        if best is None:
            fallback_p0 = [0.6 * peak, 150.0, 15.0, -0.15 * peak, 40.0, 80.0]
            best, _ = _fit_once(t, y, fallback_p0, bounds, maxfev)

        yhat = dds_func(t, *best)
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - np.mean(y))**2) + 1e-12)
        r2 = 1.0 - ss_res/ss_tot
        rmse = _rmse(y, yhat)
        
        return best, r2, rmse

    # LATE WINDOW (80-200ms)
    else:
        cfg = _late_config(peak)
        if bounds is None:
            bounds = cfg["bounds"]

        cands = []
        if p0 is not None:
            cands.append(_clip_p0(p0, bounds))
        A1b, A2b = cfg["p0"][0], cfg["p0"][3]
        for g1 in cfg["gam1_grid"]:
            for g2 in cfg["gam2_grid"]:
                for f1 in cfg["f1_grid"]:
                    for f2 in cfg["f2_grid"]:
                        for s1 in (1.0, -1.0):
                            for s2 in (1.0, -1.0):
                                cands.append([s1*A1b, g1, f1, s2*A2b, g2, f2])

        best, best_rmse = None, np.inf
        for cand in cands:
            try:
                popt, rmse = _fit_once(t, y, cand, bounds, maxfev)
                if rmse < best_rmse:
                    best, best_rmse = popt, rmse
            except Exception:
                continue

        if best is None:
            best, _ = _fit_once(t, y, cfg["p0"], bounds, maxfev)

        #yhat = dds_func(t, *best)
        #ss_res = float(np.sum((y - yhat)**2))
        #ss_tot = float(np.sum((y - np.mean(y))**2) + 1e-12)
        #r2 = 1.0 - ss_res/ss_tot
        #rmse = _rmse(y, yhat)
        yhat = dds_func(t, *best)              # model in µV
        C    = (y - yhat).mean()
        yhat = yhat + C
        res  = y - yhat
        yc   = y - y.mean()
        r2   = 1.0 - (np.var(res, ddof=1) / (np.var(yc, ddof=1) + 1e-12))
        rmse = np.sqrt(np.mean(res**2))
        return best, r2, rmse

# -------------------- helpers --------------------
def infer_site_cond(fname: str):
    n = fname.lower()
    site = "dlpfc" if "dlpfc" in n else ("m1" if "m1" in n else ("ppc" if "ppc" in n else None))
    cond = "active" if ("active" in n or "real" in n) else ("sham" if "sham" in n else None)
    if site is None or cond is None:
        raise ValueError(f"No pude inferir site/cond de: {fname}")
    return site, cond

def select_evoked_with_comment(evks, site, cond):
    cand = [e for e in evks if isinstance(e.comment, str)
            and site in e.comment.lower()
            and cond in e.comment.lower()]
    if len(cand) == 1:
        return cand[0]
    if len(cand) == 0:
        if len(evks) == 1:
            return evks[0]
        raise RuntimeError(f"Ambigüo: {len(evks)} Evokeds sin match por comment (site={site}, cond={cond}).")
    raise RuntimeError(f"Ambigüo: múltiples Evokeds ({len(cand)}) matchean comment (site={site}, cond={cond}).")

def load_evoked_eeg(fif_path: Path, tmin: float, tmax: float, site: str, cond: str):
    evks = mne.read_evokeds(str(fif_path), condition=None, verbose="ERROR")
    if isinstance(evks, mne.Evoked):
        evk = evks
    else:
        evk = select_evoked_with_comment(evks, site, cond)

    picks = mne.pick_types(evk.info, eeg=True, meg=False, eog=False, stim=False, misc=False, ecg=False, seeg=False)
    if len(picks) == 0:
        raise RuntimeError("Evoked sin canales EEG tras pick_types.")
    evk = evk.copy().pick(picks=picks, exclude=[])
    evk.crop(tmin=tmin, tmax=tmax)

    data_v = evk.get_data()         # volts
    data_uv = data_v * 1e6          # microvolts
    times = evk.times.copy()
    return data_uv, times, evk

def pick_roi_data(data_uv: np.ndarray, ch_names: list, site: str):
    roi_list = ROI_MAP.get(site, ROI_DEFAULT)
    roi_idxs = [i for i, ch in enumerate(ch_names) if ch in roi_list]
    if len(roi_idxs) == 0:
        return data_uv, ch_names, False  # fallback: all EEG (mean later)
    roi_data = data_uv[roi_idxs, :]
    roi_names = [ch_names[i] for i in roi_idxs]
    return roi_data, roi_names, True

def aggregate_signal(data_uv: np.ndarray, method: str, times_s: np.ndarray):
    if method == "mean":
        return data_uv.mean(axis=0)
    # PCA via SVD on channels x time
    X = data_uv - data_uv.mean(axis=1, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    pc1 = Vt[0, :]
    m = (times_s >= 0.0) & (times_s <= 0.03)
    sgn = np.sign(pc1[m].mean() if m.any() else pc1.mean())
    return pc1 if sgn >= 0 else -pc1

def create_joint_grand_averages_from_csv(csv_file: Path, out_dir: Path):
    """
    Create joint grand average figures from existing CSV data for both windows
    """
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    df = pd.read_csv(csv_file)
    
    # Create data structures for joint GA plotting
    ga_traces = defaultdict(list)
    ga_times = {}
    
    # Get unique windows from the data
    available_windows = df['window'].unique()
    print(f"[INFO] Found windows in CSV: {list(available_windows)}")
    
    # Process each window found in the CSV
    for window_label in available_windows:
        window_df = df[df['window'] == window_label].copy()
        
        # For joint GA, we need the actual time series data
        # Since we don't have the raw traces in CSV, we'll recreate them from parameters
        # This is a limitation - for full joint GA we need the processed data structure
        print(f"[WARN] Cannot create joint GA from CSV alone - need processed data structure for window {window_label}")
        print(f"[INFO] To create joint GA figures, run with --grand_avg and --create_joint_ga during individual processing")
    
    return False

def process_all_windows_for_joint_ga(deriv_root, out_dir, subjects, args):
    """
    Process both early and late windows to collect data for joint grand averages
    """
    all_ga_traces = defaultdict(list)
    all_ga_times = {}
    
    for window_type in ['early', 'late']:
        print(f"\n[INFO] Processing {window_type} window for joint GA...")
        
        window_config = WINDOW_CONFIG[window_type]
        tmin, tmax = window_config["tmin"], window_config["tmax"]
        win_lbl = window_config["label"]
        
        # Process this window for all subjects
        window_ga_traces, window_ga_times = process_single_window(
            deriv_root, out_dir, subjects, tmin, tmax, win_lbl, args
        )
        
        # Merge the data
        for key, traces in window_ga_traces.items():
            all_ga_traces[key].extend(traces)
        all_ga_times.update(window_ga_times)
    
    return all_ga_traces, all_ga_times

def process_single_window(deriv_root, out_dir, subjects, tmin, tmax, win_lbl, args):
    """
    Process a single window and return GA data
    """
    root = Path(deriv_root)
    ga_traces = defaultdict(list)
    ga_times = {}
    
    for sub in subjects:
        subdir = root / sub
        fif_files = sorted(subdir.glob(f"{sub}_*_ave.fif"))
        if not fif_files:
            continue

        for fif in fif_files:
            try:
                site, cond = infer_site_cond(fif.name)
                data_uv, times_s, evk = load_evoked_eeg(fif, tmin, tmax, site, cond)

                # ROI selection
                roi_data, roi_names, roi_ok = pick_roi_data(data_uv, evk.ch_names, site)

                # Aggregate within ROI
                y_uv = aggregate_signal(roi_data, method=args.agg, times_s=times_s)
                
                key = (site, cond, win_lbl)
                ga_traces[key].append(y_uv.copy())
                ga_times[key] = times_s.copy()
                
            except Exception as e:
                print(f"[ERR] {sub}/{fif.name}: {type(e).__name__}: {e}", file=sys.stderr)
                continue
    
    return ga_traces, ga_times

# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="DDS fit over EEG Evokeds with SITE-ROI aggregation (Volts->µV).")
    ap.add_argument("--deriv_root", required=False, help="Path to derivatives/mne_freedberg")
    ap.add_argument("--subject", default=None, help="ID sin 'sub-' (ej: 14). Si omitido, procesa todos los sub-*")
    ap.add_argument("--out_dir", required=True, help="Directorio de salida")
    ap.add_argument("--window", choices=["early", "late", "custom", "all"], default="early", 
                   help="Window type: early (15-80ms), late (80-200ms), custom, or all for joint GA")
    ap.add_argument("--tmin", type=float, default=None, help="inicio ventana (s) - solo para custom window")
    ap.add_argument("--tmax", type=float, default=None, help="fin ventana (s) - solo para custom window")
    ap.add_argument("--agg", choices=["pca1","mean"], default="mean", help="Agregación dentro de la ROI")
    ap.add_argument("--save_figs", action="store_true")
    ap.add_argument("--grand_avg", action="store_true",
                help="Aggregate ROI evoked traces across subjects and "
                     "plot grand-average (mean±SD) with DDS fit per site/cond/window.")
    ap.add_argument("--grand_avg_from_csv", type=str, default=None,
                   help="Create grand average from existing CSV file (skip individual processing)")
    ap.add_argument("--create_joint_ga", action="store_true",
                   help="Create joint grand average figures for selected windows")
    
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Handle grand average from CSV (skip individual processing)
    if args.grand_avg_from_csv:
        csv_file = Path(args.grand_avg_from_csv)
        if args.create_joint_ga:
            success = create_joint_grand_averages_from_csv(csv_file, out_dir)
            if not success:
                print("[INFO] For joint GA figures, please process individual files with --window all --create_joint_ga")
        return

    # Handle joint GA processing for both windows
    if args.window == "all" and args.create_joint_ga:
        if not args.deriv_root:
            raise ValueError("--deriv_root is required for joint GA processing")
            
        subjects = [f"sub-{args.subject}"] if args.subject else sorted([p.name for p in Path(args.deriv_root).glob("sub-*") if p.is_dir()])
        
        print(f"[INFO] Processing ALL windows for joint grand averages...")
        all_ga_traces, all_ga_times = process_all_windows_for_joint_ga(
            args.deriv_root, out_dir, subjects, args
        )
        
        # Create joint figures for each window
        ga_joint_dir = out_dir / "grand_averages" / "joint"
        available_windows = sorted({k[2] for k in all_ga_traces.keys()})
        
        for win_lbl in available_windows:
            out_joint = ga_joint_dir / f"GA_joint_{win_lbl.replace('-', '_')}.png"
            print(f"[INFO] Creating joint GA figure for window: {win_lbl}")
            plot_joint_grand_average(all_ga_traces, all_ga_times, win_lbl, out_joint)
        
        return

    # Normal individual file processing for single window
    if not args.deriv_root:
        raise ValueError("--deriv_root is required for individual file processing")
        
    root = Path(args.deriv_root)
    figs_dir = out_dir / "figs"
    if args.save_figs:
        figs_dir.mkdir(parents=True, exist_ok=True)

    # Set time window based on selection
    if args.window == "custom":
        if args.tmin is None or args.tmax is None:
            raise ValueError("--tmin and --tmax required for custom window")
        tmin, tmax = args.tmin, args.tmax
        win_lbl = f"{int(tmin*1000)}-{int(tmax*1000)}ms"
    else:
        window_config = WINDOW_CONFIG[args.window]
        tmin, tmax = window_config["tmin"], window_config["tmax"]
        win_lbl = window_config["label"]

    print(f"[INFO] Processing window: {win_lbl} ({tmin}-{tmax}s)")

    subjects = [f"sub-{args.subject}"] if args.subject else sorted([p.name for p in (root).glob("sub-*") if p.is_dir()])
    
    # holders for grand averages
    ga_traces = defaultdict(list)     # key: (site, cond, win_lbl) -> list of 1D arrays (µV)
    ga_times  = {}                    # key: (site, cond, win_lbl) -> time vector (s)
    
    rows = []

    for sub in subjects:
        subdir = root / sub
        fif_files = sorted(subdir.glob(f"{sub}_*_ave.fif"))
        if not fif_files:
            print(f"[WARN] No FIF in {subdir}", file=sys.stderr); continue

        for fif in fif_files:
            try:
                site, cond = infer_site_cond(fif.name)
                data_uv, times_s, evk = load_evoked_eeg(fif, tmin, tmax, site, cond)

                # ROI selection
                roi_data, roi_names, roi_ok = pick_roi_data(data_uv, evk.ch_names, site)

                # Aggregate within ROI
                y_uv = aggregate_signal(roi_data, method=args.agg, times_s=times_s)
                
                key = (site, cond, win_lbl)
                ga_traces[key].append(y_uv.copy())
                ga_times[key] = times_s.copy()
                
                # Fit DDS with enhanced optimization
                popt, r2, rmse_uv = fit_dds(times_s, y_uv, tmin=tmin, tmax=tmax)
                A1, g1, f1, A2, g2, f2 = popt

                # Validate fit quality
                yhat_uv = dds_func(times_s, *popt)
                is_valid, fit_issues = _validate_tep_fit(popt, y_uv, yhat_uv, times_s)

                rows.append({
                    "subject": sub.replace("sub-",""),
                    "site": site, "cond": cond, "window": win_lbl,
                    "tmin": tmin, "tmax": tmax,
                    "A1": A1, "gamma1": g1, "f1": f1,
                    "A2": A2, "gamma2": g2, "f2": f2,
                    "R2": r2, "RMSE_uV": rmse_uv,
                    "n_eeg": int(data_uv.shape[0]),
                    "n_roi": int(roi_data.shape[0]),
                    "roi_ok": bool(roi_ok),
                    "roi_names": ",".join(roi_names) if roi_ok else "",
                    "agg": args.agg,
                    "file": fif.name,
                    "evoked_comment": (evk.comment or ""),
                    "fit_valid": bool(is_valid),
                    "fit_issues": ";".join(fit_issues) if not is_valid else ""
                })

                if not is_valid:
                    print(f"[WARN] Poor fit quality for {sub}/{site}/{cond}: {fit_issues}")

                if args.save_figs:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(6.4, 4.4))
                    plt.plot(times_s*1e3, y_uv, label=f"Evoked ROI ({'OK' if roi_ok else 'fallback EEG mean'})", linewidth=2)
                    plt.plot(times_s*1e3, yhat_uv, "--", label="DDS fit", linewidth=2)
                    plt.xlabel("Time (ms)"); plt.ylabel("Amplitude (µV)")
                    title_roi = f"ROI={','.join(roi_names) if roi_ok else 'fallback(EEG)'}"
                    fit_status = "VALID" if is_valid else f"ISSUES:{','.join(fit_issues[:2])}"
                    plt.title(f"{sub} • {site.upper()} • {cond} • {win_lbl}\nR²={r2:.3f}  RMSE={rmse_uv:.2f}µV  {title_roi}  fit={fit_status}")
                    plt.legend(); plt.grid(alpha=0.3)
                    png = figs_dir / f"{sub}_{site}_{cond}_{win_lbl.replace('-','_')}_ddsfit_roi.png"
                    plt.tight_layout(); plt.savefig(png, dpi=300, bbox_inches="tight"); plt.close()
                    print(f"[OK] Figure: {png}")

            except Exception as e:
                print(f"[ERR] {sub}/{fif.name}: {type(e).__name__}: {e}", file=sys.stderr)
                import traceback; traceback.print_exc()

    # Save individual results
    if rows:
        df = pd.DataFrame(rows)
        per_file_csv = out_dir / f"dds_params_{win_lbl.replace('-','_')}.csv"
        df.to_csv(per_file_csv, index=False)
        print(f"[OK] Saved: {per_file_csv}")

        # Update group CSV
        group_csv = out_dir / "dds_params_group_all_windows.csv"
        if group_csv.exists():
            old = pd.read_csv(group_csv)
            keycols = ["subject","window","file"]
            old["_k"] = old[keycols].astype(str).agg("|".join, axis=1)
            df["_k"] = df[keycols].astype(str).agg("|".join, axis=1)
            grp = pd.concat([old[~old["_k"].isin(df["_k"])].drop(columns=["_k"], errors="ignore"),
                             df.drop(columns=["_k"])], ignore_index=True)
        else:
            grp = df
        grp.to_csv(group_csv, index=False)
        print(f"[OK] Updated: {group_csv}")
    else:
        print("[WARN] No rows produced.")

    # Grand average processing
    if args.grand_avg and len(ga_traces):
        ga_dir = out_dir / "grand_averages"
        ga_dir.mkdir(parents=True, exist_ok=True)
        rows_ga = []

        for key, traces in ga_traces.items():
            site_k, cond_k, win_k = key
            tvec = ga_times[key]
            Y = np.vstack(traces)
            mean_uv = Y.mean(axis=0)
            sd_uv   = Y.std(axis=0, ddof=1)

            # DDS fit to the grand-average
            popt, r2, rmse_uv = fit_dds(tvec, mean_uv, 
                                    tmin=float(win_k.split('-')[0]) / 1000.0,
                                    tmax=float(win_k.split('-')[1].replace('ms','')) / 1000.0)

            # Individual GA figure
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7.2, 4.2))
            tt_ms = tvec * 1e3
            plt.plot(tt_ms, mean_uv, label="Grand average (ROI mean)", lw=2)
            plt.fill_between(tt_ms, mean_uv - sd_uv, mean_uv + sd_uv,
                         alpha=0.25, label="±1 SD")
            yhat = dds_func(tvec, *popt)
            plt.plot(tt_ms, yhat, "--", lw=2, label="DDS fit (mean)")
            plt.xlabel("Time (ms)"); plt.ylabel("Amplitude (µV)")
            plt.title(f"{site_k.upper()} • {cond_k} • {win_k}\n"
                  f"R²={r2:.3f}  RMSE={rmse_uv:.2f} µV")
            plt.grid(alpha=0.3); plt.legend(frameon=False)
            plt.tight_layout()
            fig_path = ga_dir / f"GA_{site_k}_{cond_k}_{win_k.replace('-','_')}.png"
            plt.savefig(fig_path, dpi=600, bbox_inches="tight"); plt.close()
            print(f"[OK] Grand-average figure saved: {fig_path}")

            # store GA fit parameters
            A1, g1, f1, A2, g2, f2 = popt
            rows_ga.append({
                "site": site_k, "cond": cond_k, "window": win_k,
                "A1": A1, "gamma1": g1, "f1": f1, "A2": A2, "gamma2": g2, "f2": f2,
                "R2": r2, "RMSE_uV": rmse_uv, "n_subjects": Y.shape[0]
            })

        # CSV with GA parameters
        if rows_ga:
            ga_csv = ga_dir / "dds_params_grand_averages.csv"
            pd.DataFrame(rows_ga).to_csv(ga_csv, index=False)
            print(f"[OK] Grand-average params saved: {ga_csv}")

    # Joint grand average figures for single window processing
    if args.create_joint_ga and len(ga_traces):
        ga_joint_dir = out_dir / "grand_averages" / "joint"
        out_joint = ga_joint_dir / f"GA_joint_{win_lbl.replace('-', '_')}.png"
        print(f"[INFO] Creating joint GA figure for window: {win_lbl}")
        plot_joint_grand_average(ga_traces, ga_times, win_lbl, out_joint)
    
if __name__ == "__main__":
    main()
