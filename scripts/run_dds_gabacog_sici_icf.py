#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path

import mne  # en tu entorno funciona
from configs.gabacog_sici_icf import WINDOWS, BASELINE, ROI
from scripts.dds_model import fit_dds_double, fit_dds_single  # ya existe en tu repo

def r2_score(y, yhat):
    y = np.asarray(y); yhat = np.asarray(yhat)
    ss_res = np.nansum((y - yhat)**2)
    ss_tot = np.nansum((y - np.nanmean(y))**2) + 1e-12
    return 1 - ss_res/ss_tot

def roi_average(evoked, ch_names):
    picks = [evoked.ch_names.index(ch) for ch in ch_names if ch in evoked.ch_names]
    if len(picks) == 0:
        return None
    data = evoked.data[picks, :].mean(axis=0)
    return data

def fit_one_trace(t, y, w0, w1, model="double", fmax=45.0, t0=0.0):
    mask = (t >= w0) & (t <= w1)
    tt = t[mask]
    yy = y[mask]

    if model == "double":
        params, yhat = fit_dds_double(tt, yy, fmax=fmax, t0=t0, return_yhat=True)
    else:
        params, yhat = fit_dds_single(tt, yy, fmax=fmax, t0=t0, return_yhat=True)

    r2 = r2_score(yy, yhat)
    return params, r2

def main():
    outdir = Path("derivatives") / "dds_gabacog"
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(outdir / "manifest_gabacog_sici_icf.csv")

    rows_ch = []
    rows_roi = []

    for _, r in manifest.iterrows():
        epochs = mne.read_epochs(r.fif_path, preload=True, verbose="ERROR")

        # baseline explícito
        epochs.apply_baseline(BASELINE)

        evk = epochs.average()

        t = evk.times
        t0 = 0.0

        for (w0, w1, wlab) in WINDOWS[r.protocol]:
            # ---- channel-wise (primario)
            for ch_i, ch in enumerate(evk.ch_names):
                y = evk.data[ch_i, :]
                try:
                    params, r2 = fit_one_trace(t, y, w0, w1, model="double", fmax=45.0, t0=t0)
                except Exception:
                    continue

                rows_ch.append({
                    "subject_id": r.subject_id,
                    "protocol": r.protocol,
                    "group": r.group,
                    "window": wlab,
                    "channel": ch,
                    "r2": r2,
                    **params
                })

            # ---- ROI (secundario)
            for roi_name, roi_chs in ROI.items():
                yroi = roi_average(evk, roi_chs)
                if yroi is None:
                    continue
                try:
                    params, r2 = fit_one_trace(t, yroi, w0, w1, model="double", fmax=45.0, t0=t0)
                except Exception:
                    continue
                rows_roi.append({
                    "subject_id": r.subject_id,
                    "protocol": r.protocol,
                    "group": r.group,
                    "window": wlab,
                    "roi": roi_name,
                    "r2": r2,
                    **params
                })

    df_ch  = pd.DataFrame(rows_ch)
    df_roi = pd.DataFrame(rows_roi)

    df_ch.to_csv(outdir / "dds_params_channelwise.csv", index=False)
    df_roi.to_csv(outdir / "dds_params_roi.csv", index=False)

    print("Saved:", outdir / "dds_params_channelwise.csv")
    print("Saved:", outdir / "dds_params_roi.csv")

if __name__ == "__main__":
    main()

