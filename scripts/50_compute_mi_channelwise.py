from pathlib import Path
import numpy as np
import pandas as pd
import mne

DDS_ALL = Path("outputs/exports/dds/dds_params_all.csv")
EVOKEDS_ROOT = Path("outputs/exports/evokeds")
OUT = Path("outputs/features/info_mutual_information_channelwise.csv")

RESAMPLE_N = 128

def zscore(x):
    x = np.asarray(x, float)
    if x.size == 0: return x
    mu = np.nanmean(x); sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return x - mu
    return (x - mu) / sd

def resample_to_n(x, n):
    x = np.asarray(x, float)
    if x.size == 0: return x
    if x.size == n: return x
    t_old = np.linspace(0.0, 1.0, x.size)
    t_new = np.linspace(0.0, 1.0, n)
    return np.interp(t_new, t_old, x)

def fd_bins(x, max_bins=128, fallback=32):
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 10: return fallback
    q75, q25 = np.percentile(x, [75,25])
    iqr = q75-q25
    if iqr <= 1e-12: return fallback
    bw = 2*iqr*(n**(-1/3))
    if bw <= 1e-12: return fallback
    bins = int(np.ceil((x.max()-x.min())/bw))
    return int(np.clip(bins, 8, max_bins))

def mi_hist_pairs(x, y, bx, by):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.size != y.size or x.size < 10: return np.nan
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]; y = y[mask]
    if x.size < 10: return np.nan
    hxy, _, _ = np.histogram2d(x, y, bins=[bx, by])
    s = hxy.sum()
    if s <= 0: return np.nan
    pxy = hxy / s
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mi = 0.0
    for i in range(pxy.shape[0]):
        for j in range(pxy.shape[1]):
            pij = pxy[i, j]
            if pij > 0 and px[i] > 0 and py[j] > 0:
                mi += pij * np.log2(pij / (px[i] * py[j]))
    return float(mi)

def load_evoked(subject, fname):
    f = EVOKEDS_ROOT / f"sub-{subject:02d}" / fname
    ev = mne.read_evokeds(str(f), verbose="ERROR")[0]
    return ev

def crop_idx(times, tmin, tmax):
    idx = np.where((times >= tmin) & (times <= tmax))[0]
    return idx

def main():
    df = pd.read_csv(DDS_ALL)

    # necesitamos ventanas early/late por combinación
    rows = []
    for (sub, site, cond), g in df.groupby(["subject","site","cond"]):
        if not {"early","late"} <= set(g["window"]):
            continue
        rE = g[g["window"]=="early"].iloc[0]
        rL = g[g["window"]=="late"].iloc[0]
        ev = load_evoked(int(sub), str(rE["file"]))

        # solo EEG
        picks = mne.pick_types(ev.info, eeg=True, meg=False)
        ch_names = [ev.ch_names[i] for i in picks]
        data = ev.data[picks, :]  # (n_ch, n_times)

        idxE = crop_idx(ev.times, float(rE["tmin"]), float(rE["tmax"]))
        idxL = crop_idx(ev.times, float(rL["tmin"]), float(rL["tmax"]))

        for ci, ch in enumerate(ch_names):
            x = data[ci, idxE] if idxE.size else np.array([])
            y = data[ci, idxL] if idxL.size else np.array([])
            if x.size == 0 or y.size == 0:
                mi = np.nan
                reason = "crop_empty"
            else:
                x = zscore(x); y = zscore(y)
                xN = resample_to_n(x, RESAMPLE_N)
                yN = resample_to_n(y, RESAMPLE_N)
                bx = fd_bins(xN); by = fd_bins(yN)
                mi = mi_hist_pairs(xN, yN, bx, by)
                reason = "ok" if np.isfinite(mi) else "mi_nan"
            rows.append({
                "subject": int(sub),
                "site": site,
                "cond": cond,
                "channel": ch,
                "mi_early_late_bits": mi,
                "reason": reason,
                "resample_n": RESAMPLE_N,
            })

        print(f"[{int(sub):02d} {site} {cond}] channels={len(ch_names)} done")

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print("[OK] wrote:", OUT, "rows=", len(out))
    print("MI finite %:", (out["mi_early_late_bits"].notna() & np.isfinite(out["mi_early_late_bits"])).mean()*100)
    print("Reason counts:\n", out["reason"].value_counts().head(10))
if __name__ == "__main__":
    main()
