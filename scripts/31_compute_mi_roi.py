"""
31_compute_mutual_information_debug.py

Versión con trazas para diagnosticar por qué MI sale NaN.
Genera:
- outputs/features/info_mutual_information_debug.csv
- outputs/features/mi_debug_summary.txt
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import mne


DDS_ALL = Path("outputs/exports/dds/dds_params_all.csv")
EVOKEDS_ROOT = Path("outputs/exports/evokeds")
OUT_MI = Path("outputs/features/info_mutual_information_debug.csv")
OUT_SUM = Path("outputs/features/mi_debug_summary.txt")

RESAMPLE_N = 128


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    if x.size == 0:
        return x
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return x - mu
    return (x - mu) / sd


def resample_to_n(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, float)
    if x.size == 0:
        return x
    if x.size == n:
        return x
    t_old = np.linspace(0.0, 1.0, x.size)
    t_new = np.linspace(0.0, 1.0, n)
    return np.interp(t_new, t_old, x)


def freedman_diaconis_bins(x: np.ndarray, max_bins: int = 128, fallback: int = 32) -> int:
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 10:
        return fallback
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr <= 1e-12:
        return fallback
    bw = 2 * iqr * (n ** (-1 / 3))
    if bw <= 1e-12:
        return fallback
    bins = int(np.ceil((x.max() - x.min()) / bw))
    return int(np.clip(bins, 8, max_bins))


def mutual_information_hist_pairs(x: np.ndarray, y: np.ndarray, bins_x: int, bins_y: int) -> tuple[float, str]:
    # devuelve (mi, reason_if_nan)
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    if x.size != y.size:
        return np.nan, "len_mismatch_after_resample"
    if x.size < 10:
        return np.nan, "too_few_samples"

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if x.size < 10:
        return np.nan, "too_few_finite_pairs"

    hxy, _, _ = np.histogram2d(x, y, bins=[bins_x, bins_y])
    s = hxy.sum()
    if s <= 0:
        return np.nan, "hist2d_empty"

    pxy = hxy / s
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)

    mi = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            pij = pxy[i, j]
            if pij > 0 and px[i] > 0 and py[j] > 0:
                mi += pij * np.log2(pij / (px[i] * py[j]))

    if not np.isfinite(mi):
        return np.nan, "mi_not_finite"
    return float(mi), ""


def load_evoked(subject: int, fname: str) -> mne.Evoked:
    f = EVOKEDS_ROOT / f"sub-{subject:02d}" / fname
    if not f.exists():
        raise FileNotFoundError(f"Evoked not found: {f}")
    return mne.read_evokeds(str(f), verbose="ERROR")[0]


def roi_trace_mean(ev: mne.Evoked, roi_names: str) -> tuple[np.ndarray, list[str]]:
    chs = [c.strip() for c in str(roi_names).split(",") if c.strip()]
    present = [c for c in chs if c in ev.ch_names]
    if not present:
        return np.array([]), present
    return ev.copy().pick(present).data.mean(axis=0), present


def crop(ev: mne.Evoked, tr: np.ndarray, tmin: float, tmax: float) -> tuple[np.ndarray, int]:
    idx = np.where((ev.times >= tmin) & (ev.times <= tmax))[0]
    if idx.size == 0:
        return np.array([]), 0
    return tr[idx], int(idx.size)


def main():
    df = pd.read_csv(DDS_ALL)

    keys = ["subject", "site", "cond"]
    rows = []

    for (sub, site, cond), g in df.groupby(keys):
        if not {"early", "late"} <= set(g["window"]):
            continue

        rE = g[g["window"] == "early"].iloc[0]
        rL = g[g["window"] == "late"].iloc[0]

        rec = {
            "subject": int(sub),
            "site": site,
            "cond": cond,
            "file": str(rE.get("file", "")),
            "roi_names": str(rE.get("roi_names", "")),
            "tmin_early": float(rE.get("tmin", np.nan)),
            "tmax_early": float(rE.get("tmax", np.nan)),
            "tmin_late": float(rL.get("tmin", np.nan)),
            "tmax_late": float(rL.get("tmax", np.nan)),
            "reason": "",
        }

        try:
            ev = load_evoked(int(sub), rec["file"])
            rec["evoked_nchan"] = int(ev.data.shape[0])
            rec["evoked_nt"] = int(ev.data.shape[1])
            rec["time0"] = float(ev.times[0])
            rec["time_end"] = float(ev.times[-1])
            rec["sfreq_est"] = float(1.0 / np.median(np.diff(ev.times)))

            tr, present = roi_trace_mean(ev, rec["roi_names"])
            rec["roi_present_n"] = int(len(present))
            rec["roi_present"] = ",".join(present)

            if tr.size == 0:
                rec["reason"] = "roi_empty"
                rec["mi_early_late_bits"] = np.nan
                rows.append(rec)
                continue

            x, nx = crop(ev, tr, rec["tmin_early"], rec["tmax_early"])
            y, ny = crop(ev, tr, rec["tmin_late"], rec["tmax_late"])
            rec["n_early_samples"] = int(nx)
            rec["n_late_samples"] = int(ny)

            if x.size == 0:
                rec["reason"] = "crop_early_empty"
                rec["mi_early_late_bits"] = np.nan
                rows.append(rec)
                continue
            if y.size == 0:
                rec["reason"] = "crop_late_empty"
                rec["mi_early_late_bits"] = np.nan
                rows.append(rec)
                continue

            x = zscore(x)
            y = zscore(y)

            xN = resample_to_n(x, RESAMPLE_N)
            yN = resample_to_n(y, RESAMPLE_N)
            rec["resample_n"] = int(RESAMPLE_N)
            rec["xN_len"] = int(xN.size)
            rec["yN_len"] = int(yN.size)

            if xN.size == 0 or yN.size == 0:
                rec["reason"] = "resample_empty"
                rec["mi_early_late_bits"] = np.nan
                rows.append(rec)
                continue

            bx = freedman_diaconis_bins(xN)
            by = freedman_diaconis_bins(yN)
            rec["bins_early"] = int(bx)
            rec["bins_late"] = int(by)

            mi, reason = mutual_information_hist_pairs(xN, yN, bx, by)
            rec["mi_early_late_bits"] = mi
            rec["reason"] = reason or ("ok" if np.isfinite(mi) else "mi_nan_unknown")

        except Exception as e:
            rec["mi_early_late_bits"] = np.nan
            rec["reason"] = "exception"
            rec["error"] = repr(e)

        rows.append(rec)

    out = pd.DataFrame(rows)
    OUT_MI.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_MI, index=False)

    # resumen
    lines = []
    lines.append(f"Rows: {len(out)}")
    lines.append(f"MI finite: {(out['mi_early_late_bits'].notna() & np.isfinite(out['mi_early_late_bits'])).mean()*100:.1f}%")
    lines.append("Reason counts:")
    lines.append(out["reason"].value_counts(dropna=False).to_string())
    lines.append("\nSample of non-ok rows:")
    lines.append(out[out["reason"] != "ok"].head(20).to_string(index=False))

    OUT_SUM.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote: {OUT_MI}")
    print(f"[OK] wrote: {OUT_SUM}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

