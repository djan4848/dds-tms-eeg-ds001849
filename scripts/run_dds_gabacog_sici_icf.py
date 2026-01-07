#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path

import mne
from configs.gabacog_sici_icf import WINDOWS, BASELINE, ROI
from scripts.dds_model import fit_dds, roi_average  # usar la del módulo (no redefinir)

# -------------------------
# Config
# -------------------------
KEEP_GROUPS = {"CTL", "TOC"}   # excluye NA
H_FREQ = 100.0                # coherente con tu prepro
TWO_COMPONENTS = True         # DDS doble para ambos protocolos (como venimos usando)
VERBOSE_EVERY = 10            # log cada N ficheros

# -------------------------
# Utils
# -------------------------
def fit_one_trace(t, y, w0, w1, h_freq=H_FREQ, two_components=TWO_COMPONENTS):
    """
    Fit DDS on full time vector but within [w0, w1].
    Returns params dict (includes R2, RMSE, fs, fmax_used, etc.) and yhat.
    """
    params, yhat = fit_dds(
        time_s=t,
        signal=y,
        tmin=w0,
        tmax=w1,
        two_components=two_components,
        t0=0.0,
        h_freq=h_freq,
    )
    return params, yhat

def main():
    outdir = Path("derivatives") / "dds_gabacog"
    outdir.mkdir(parents=True, exist_ok=True)

    manifest_path = outdir / "manifest_gabacog_sici_icf.csv"
    manifest = pd.read_csv(manifest_path)

    # --- filtro: solo CTL/TOC
    n0 = len(manifest)
    manifest = manifest[manifest["group"].isin(KEEP_GROUPS)].copy()
    print(f"[RUN_DDS] Loaded manifest: {n0} rows | Using CTL/TOC only: {len(manifest)} rows")

    rows_ch, rows_roi = [], []

    # contadores de robustez/depuración
    n_files = 0
    n_fail_file = 0
    n_fail_fit_ch = 0
    n_fail_fit_roi = 0

    for i, r in manifest.iterrows():
        n_files += 1
        if (n_files % VERBOSE_EVERY) == 0:
            print(f"[RUN_DDS] Processing file {n_files}/{len(manifest)}: sub {r.subject_id} | {r.protocol} | {r.group}")

        # ---- leer epochs y baseline
        try:
            epochs = mne.read_epochs(r.fif_path, preload=True, verbose="ERROR")
            epochs.apply_baseline(BASELINE)
            evk = epochs.average()
        except Exception as e:
            n_fail_file += 1
            print(f"[WARN] Failed to read/average: {r.fif_path} | {repr(e)}")
            continue

        t = evk.times

        for (w0, w1, wlab) in WINDOWS[r.protocol]:

            # -------------------------
            # Channel-wise (primario)
            # -------------------------
            for ch_i, ch in enumerate(evk.ch_names):
                y = evk.data[ch_i, :]
                try:
                    params, _ = fit_one_trace(t, y, w0, w1, h_freq=H_FREQ, two_components=True)
                except Exception:
                    n_fail_fit_ch += 1
                    continue

                rows_ch.append({
                    "subject_id": r.subject_id,
                    "protocol": r.protocol,
                    "group": r.group,
                    "window": wlab,
                    "channel": ch,
                    "tmin": w0,
                    "tmax": w1,
                    "h_freq": H_FREQ,
                    "fs": params.get("fs", np.nan),
                    "fmax_used": params.get("fmax_used", np.nan),
                    "two_components": True,
                    **params,
                })

            # -------------------------
            # ROI (secundario)
            # -------------------------
            for roi_name, roi_chs in ROI.items():
            # filtra a canales presentes para evitar que roi_average reviente
                roi_present = [ch for ch in roi_chs if ch in evk.ch_names]
                if len(roi_present) < 2:
                    continue  # demasiado pequeño / vacío
                yroi = roi_average(evk, roi_present)
                if yroi is None:
                    continue
                try:
                    params, _ = fit_one_trace(t, yroi, w0, w1, h_freq=H_FREQ, two_components=True)
                except Exception:
                    n_fail_fit_roi += 1
                    continue

                rows_roi.append({
                    "subject_id": r.subject_id,
                    "protocol": r.protocol,
                    "group": r.group,
                    "window": wlab,
                    "roi": roi_name,
                    "tmin": w0,
                    "tmax": w1,
                    "h_freq": H_FREQ,
                    "fs": params.get("fs", np.nan),
                    "fmax_used": params.get("fmax_used", np.nan),
                    "two_components": True,
                    **params,
                })

    df_ch  = pd.DataFrame(rows_ch)
    df_roi = pd.DataFrame(rows_roi)

    out_ch  = outdir / "dds_params_channelwise.csv"
    out_roi = outdir / "dds_params_roi.csv"
    df_ch.to_csv(out_ch, index=False)
    df_roi.to_csv(out_roi, index=False)

    print("\n[RUN_DDS] Done.")
    print(f"[RUN_DDS] Files processed: {n_files} | failed read/avg: {n_fail_file}")
    print(f"[RUN_DDS] Fit failures: channel-wise={n_fail_fit_ch} | ROI={n_fail_fit_roi}")
    print("Saved:", out_ch)
    print("Saved:", out_roi)

if __name__ == "__main__":
    main()

