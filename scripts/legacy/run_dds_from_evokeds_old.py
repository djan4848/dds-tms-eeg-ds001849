#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DDS fitter for TMS-EEG Evokeds with SITE-ROI aggregation (fallback to EEG mean).

Addresses:
- EEG-only picking (no MEG/EOG/etc.)
- Evoked selection by comment matching (site/cond) to avoid ambiguity
- Units: Volts -> microvolts for outputs/plots
- Spatial aggregation by ROI per site (NOT all channels). If ROI channels not found -> fallback to mean of all EEG.

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

# -------------------- ROI config --------------------
ROI_DEFAULT = ["C3", "FC1", "CP1", "FC5"]
ROI_MAP = {
    "m1":    ["FC3", "C3", "CP3"], #left M1
    "dlpfc": ["F4", "Fp2", "F8", "FC2", "FC6"], #right dlpfc
    "ppc":   ["P3", "P7", "CP1", "CP5", "Pz"]   #left ppc
}

# -------------------- DDS model --------------------
def dds_func(t, A1, gamma1, f1, A2, gamma2, f2):
    return (
        A1 * np.exp(-gamma1 * t) * np.sin(2 * np.pi * f1 * t) +
        A2 * np.exp(-gamma2 * t) * np.sin(2 * np.pi * f2 * t)
    )

# --- WINDOW-SPECIFIC PARAMETERIZATION FOR DDS FIT ---
# Paste these definitions in your script, replacing the existing init_by_window()
# and fit_dds() implementations.

import numpy as np
from scipy.optimize import curve_fit
def _ensure_in_bounds(p, bounds, eps=1e-6):
    lb, ub = np.asarray(bounds[0], float), np.asarray(bounds[1], float)
    p = np.asarray(p, float)
    # clip p strictly inside bounds to avoid "x0 is infeasible"
    return np.minimum(ub - eps, np.maximum(lb + eps, p)).tolist()

def window_config(tmin: float, tmax: float, peak: float):
    """
    Early window (0–100 ms) retuned:
      - A1 seed = 1.10*peak
      - γ1 grid/prior: {220, 260, 180} s^-1  (bounds [140, 320])
      - A2 seed = -0.25*peak (negative priority)
      - γ2 grid/prior: {20, 24, 28, 16} s^-1 (bounds [12, 35])
      - f1 grid/prior: {10, 9, 11} Hz       (bounds [8, 12])
      - f2 grid/prior: {75, 70, 80, 65} Hz  (bounds [60, 85])

    Late window (100–200 ms) unchanged.
    """
    if tmax <= 0.100:  # EARLY
        return {
            # bounds: [A1, γ1, f1,  A2, γ2, f2]
            "bounds": ([-1e4, 140.0, 8.0,   -1e4, 12.0, 60.0],
                       [ 1e4, 320.0, 12.0,   1e4, 35.0, 85.0]),
            # p0 = [A1, γ1, f1,  A2, γ2, f2]
            "p0": [1.10*peak, 220.0, 10.0, -0.25*peak, 22.0, 75.0],
            "gam1_grid": [220.0, 260.0, 180.0],
            "gam2_grid": [20.0, 24.0, 28.0, 16.0],
            "f1_grid":   [10.0, 9.0, 11.0],
            "f2_grid":   [75.0, 70.0, 80.0, 65.0],
        }
    else:  # LATE (unchanged)
        return {
            "bounds": ([-1e4, 1.0, 4.0,  -1e4, 1.0, 50.0],
                       [ 1e4, 12.0, 12.0,  1e4, 12.0, 65.0]),
            "p0": [0.60*peak, 3.0, 8.0, 0.60*peak, 3.0, 58.0],
            "gam1_grid": [2.0, 4.0, 6.0],
            "gam2_grid": [2.0, 4.0, 6.0],
            "f1_grid":   [6.0, 8.0, 10.0, 12.0],
            "f2_grid":   [55.0, 60.0, 62.0],
        }
# ---------- Utilities ----------

# ---------------- Utils ----------------
# ---------------- Utils ----------------
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

# ---------------- Early window parametrization (0–100 ms) ----------------
def _early_dynamic_bounds(peak):
    # Dynamic amplitude bounds (A1 mostly positive; A2 limited magnitude; sign handled in grid)
    a1_lo = max(1e-3, 0.5 * peak)
    a1_hi = max(a1_lo + 1e-3, 2.0 * peak)
    a2_lo = -0.4 * peak
    a2_hi =  0.4 * peak
    # [A1, γ1, f1,  A2, γ2, f2]
    return ([a1_lo, 120.0, 8.0,   a2_lo, 10.0, 56.0],
            [a1_hi, 400.0, 12.0,  a2_hi, 45.0, 70.0])

def _early_candidates(peak):
    # Seeds/grids oriented a corregir el desfasaje sin fase explícita:
    #   - γ1 más amplio (pico más lento/rápido)
    #   - γ2 más suave (rizado menos amortiguado)
    #   - f2 centrado 56–70 Hz para ajustar el segundo ciclo
    A1_seed = 1.10 * peak
    A2_mag  = 0.15 * peak

    gam1_grid = [140.0, 180.0, 220.0, 260.0, 300.0, 340.0, 380.0]
    gam2_grid = [18.0, 22.0, 26.0, 30.0, 34.0, 38.0, 42.0]
    f1_grid   = [9.0, 10.0, 11.0]
    f2_grid   = [56.0, 58.0, 60.0, 62.0, 64.0, 66.0, 68.0, 70.0]

    cands = []
    for s2 in (-1.0, +1.0):  # prioriza A2 negativo por orden
        for g1 in gam1_grid:
            for g2 in gam2_grid:
                for f1 in f1_grid:
                    for f2 in f2_grid:
                        cands.append([A1_seed, g1, f1, s2*A2_mag, g2, f2])
    return cands

def _select_best_early(t, y, bounds, candidates, maxfev):
    # Weighted selection: 70% global RMSE + 30% sub-window RMSE (5–25 ms)
    sub_mask = (t >= 0.005) & (t <= 0.025)
    best = None
    best_score = np.inf
    for p0 in candidates:
        try:
            popt, rmse_g = _fit_once(t, y, p0, bounds, maxfev)
            yhat = dds_func(t, *popt)
            rmse_sub = _rmse(y[sub_mask], yhat[sub_mask]) if sub_mask.any() else rmse_g
            score = 0.7 * rmse_g + 0.3 * rmse_sub
            if score < best_score:
                best, best_score = popt, score
        except Exception:
            continue
    return best

# ---------------- Late window parametrization (100–200 ms) ----------------
def _late_config(peak):
    return {
        "bounds": ([-1e4, 1.0, 4.0,  -1e4, 1.0, 50.0],
                   [ 1e4, 12.0, 12.0,  1e4, 12.0, 65.0]),
        "p0": [0.60*peak, 3.0, 8.0, 0.60*peak, 3.0, 58.0],
        "gam1_grid": [2.0, 4.0, 6.0],
        "gam2_grid": [2.0, 4.0, 6.0],
        "f1_grid":   [6.0, 8.0, 10.0, 12.0],
        "f2_grid":   [55.0, 60.0, 62.0],
    }

# ---------------- Main fitting ----------------
def fit_dds(time_s, y, tmin, tmax, p0=None, bounds=None, maxfev=120000):
    """
    DDS fit (sin fase).
    Early (<=100 ms): dynamic bounds + broad grid + weighted sub-window selection + jitter
                      con énfasis en {A2, γ2, f2}.
    Late  (>100 ms): grid multistart estándar.
    """
    t = np.asarray(time_s, float)
    y = np.asarray(y, float)
    peak = float(np.nanmax(np.abs(y))) if np.isfinite(y).any() else 1.0

    # EARLY
    if tmax <= 0.100:
        if bounds is None:
            bounds = _early_dynamic_bounds(peak)

        candidates = _early_candidates(peak)
        if p0 is not None:
            candidates.insert(0, p0)

        best = _select_best_early(t, y, bounds, candidates, maxfev)

        # Jitter local (foco en A2, γ2, f2) para afinar desfasaje sin fase explícita
        if best is not None:
            best = np.asarray(best, float)
            scales = np.array([0.08, 0.08, 0.04, 0.25, 0.20, 0.08])  # [A1,γ1,f1,A2,γ2,f2]
            jitters = []
            for _ in range(36):
                noise = (np.random.rand(6) - 0.5) * 2.0 * scales
                p0j = best * (1.0 + noise)
                jitters.append(p0j.tolist())
            jb = _select_best_early(t, y, bounds, jitters, maxfev)
            if jb is not None:
                # keep better weighted score
                def score(p):
                    yhat = dds_func(t, *p)
                    rmse_g = _rmse(y, yhat)
                    sub = (t >= 0.005) & (t <= 0.025)
                    rmse_s = _rmse(y[sub], yhat[sub]) if sub.any() else rmse_g
                    return 0.7 * rmse_g + 0.3 * rmse_s
                if score(jb) < score(best):
                    best = jb

        if best is None:
            seed = [1.10*peak, 220.0, 10.0, -0.15*peak, 26.0, 62.0]
            best, _ = _fit_once(t, y, seed, bounds, maxfev)

        yhat = dds_func(t, *best)
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - np.mean(y))**2) + 1e-12)
        r2 = 1.0 - ss_res/ss_tot
        rmse = _rmse(y, yhat)
        return best, r2, rmse

    # LATE
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

    yhat = dds_func(t, *best)
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2) + 1e-12)
    r2 = 1.0 - ss_res/ss_tot
    rmse = _rmse(y, yhat)
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
    # intersect requested ROI with available channel names (case-sensitive as in MNE)
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
    # sign alignment using 0–30 ms mean if available
    m = (times_s >= 0.0) & (times_s <= 0.03)
    sgn = np.sign(pc1[m].mean() if m.any() else pc1.mean())
    return pc1 if sgn >= 0 else -pc1


# -------------------- main --------------------
def main():
    ap = argparse.ArgumentParser(description="DDS fit over EEG Evokeds with SITE-ROI aggregation (Volts->µV).")
    ap.add_argument("--deriv_root", required=True, help="Path to derivatives/mne_freedberg")
    ap.add_argument("--subject", default=None, help="ID sin 'sub-' (ej: 14). Si omitido, procesa todos los sub-*")
    ap.add_argument("--out_dir", required=True, help="Directorio de salida")
    ap.add_argument("--tmin", type=float, default=0.0, help="inicio ventana (s)")
    ap.add_argument("--tmax", type=float, default=0.1, help="fin ventana (s)")
    ap.add_argument("--agg", choices=["pca1","mean"], default="mean", help="Agregación dentro de la ROI")
    ap.add_argument("--save_figs", action="store_true")
    args = ap.parse_args()

    root = Path(args.deriv_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    figs_dir = out_dir / "figs"
    if args.save_figs:
        figs_dir.mkdir(parents=True, exist_ok=True)

    subjects = [f"sub-{args.subject}"] if args.subject else sorted([p.name for p in (root).glob("sub-*") if p.is_dir()])
    rows = []
    win_lbl = f"{int(args.tmin*1000)}-{int(args.tmax*1000)}ms"

    for sub in subjects:
        subdir = root / sub
        fif_files = sorted(subdir.glob(f"{sub}_*_ave.fif"))
        if not fif_files:
            print(f"[WARN] No FIF in {subdir}", file=sys.stderr); continue

        for fif in fif_files:
            try:
                site, cond = infer_site_cond(fif.name)  # site in {'m1','dlpfc','ppc'}, cond in {'active','sham'}
                data_uv, times_s, evk = load_evoked_eeg(fif, args.tmin, args.tmax, site, cond)

                # ROI selection (fallback to all EEG if none of the desired channels exists)
                roi_data, roi_names, roi_ok = pick_roi_data(data_uv, evk.ch_names, site)

                # Aggregate within ROI
                y_uv = aggregate_signal(roi_data, method=args.agg, times_s=times_s)

                # Fit DDS
                #p0 = init_by_window(args.tmin, args.tmax)
                popt, r2, rmse_uv = fit_dds(times_s, y_uv, tmin=args.tmin, tmax=args.tmax)
                A1, g1, f1, A2, g2, f2 = popt

                rows.append({
                    "subject": sub.replace("sub-",""),
                    "site": site, "cond": cond, "window": win_lbl,
                    "tmin": args.tmin, "tmax": args.tmax,
                    "A1": A1, "gamma1": g1, "f1": f1,
                    "A2": A2, "gamma2": g2, "f2": f2,
                    "R2": r2, "RMSE_uV": rmse_uv,
                    "n_eeg": int(data_uv.shape[0]),
                    "n_roi": int(roi_data.shape[0]),
                    "roi_ok": bool(roi_ok),
                    "roi_names": ",".join(roi_names) if roi_ok else "",
                    "agg": args.agg,
                    "file": fif.name,
                    "evoked_comment": (evk.comment or "")
                })

                if args.save_figs:
                    import matplotlib.pyplot as plt
                    yhat_uv = dds_func(times_s, *popt)
                    plt.figure(figsize=(6.4, 4.4))
                    plt.plot(times_s*1e3, y_uv, label=f"Evoked ROI ({'OK' if roi_ok else 'fallback EEG mean'})", linewidth=2)
                    plt.plot(times_s*1e3, yhat_uv, "--", label="DDS fit", linewidth=2)
                    plt.xlabel("Time (ms)"); plt.ylabel("Amplitude (µV)")
                    title_roi = f"ROI={','.join(roi_names) if roi_ok else 'fallback(EEG)'}"
                    plt.title(f"{sub} • {site.upper()} • {cond} • {win_lbl}\nR²={r2:.3f}  RMSE={rmse_uv:.2f}µV  {title_roi}  agg={args.agg}")
                    plt.legend(); plt.grid(alpha=0.3)
                    png = figs_dir / f"{sub}_{site}_{cond}_{win_lbl.replace('-','_')}_ddsfit_roi.png"
                    plt.tight_layout(); plt.savefig(png, dpi=300, bbox_inches="tight"); plt.close()
                    print(f"[OK] Figure: {png}")

            except Exception as e:
                print(f"[ERR] {sub}/{fif.name}: {type(e).__name__}: {e}", file=sys.stderr)
                import traceback; traceback.print_exc()

    if rows:
        df = pd.DataFrame(rows)
        per_file_csv = out_dir / f"dds_params_{win_lbl.replace('-','_')}.csv"
        df.to_csv(per_file_csv, index=False)
        print(f"[OK] Saved: {per_file_csv}")

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

if __name__ == "__main__":
    main()

