"""
30_compute_entropy.py

Calcula entropía de Shannon sobre la traza ROI (media de canales ROI)
para cada fila en outputs/exports/dds/dds_params_all.csv.

Salida:
- outputs/features/info_entropy.csv
- outputs/features/dds_plus_entropy.csv

Notas:
- Se usa la misma definición de ROI (roi_names) y ventanas (tmin/tmax) que DDS.
- La entropía se calcula sobre amplitudes z-scored dentro de la ventana.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import mne


DDS_ALL = Path("outputs/exports/dds/dds_params_all.csv")
EVOKEDS_ROOT = Path("outputs/exports/evokeds")
OUT_ENT = Path("outputs/features/info_entropy.csv")
OUT_MERGED = Path("outputs/features/dds_plus_entropy.csv")


def freedman_diaconis_bins(x: np.ndarray, max_bins: int = 256, fallback: int = 64) -> int:
    """Número de bins por regla Freedman–Diaconis, con fallbacks robustos."""
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
    if bins < 8:
        bins = 8
    if bins > max_bins:
        bins = max_bins
    return bins


def shannon_entropy_hist(x: np.ndarray, bins: int) -> float:
    """Entropía de Shannon (base 2) a partir de histograma."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    hist, _ = np.histogram(x, bins=bins, density=False)
    p = hist.astype(float)
    s = p.sum()
    if s <= 0:
        return np.nan
    p /= s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def load_evoked_for_row(row: pd.Series) -> mne.Evoked:
    subj = int(row["subject"])
    sub_dir = EVOKEDS_ROOT / f"sub-{subj:02d}"
    f = sub_dir / str(row["file"])
    if not f.exists():
        raise FileNotFoundError(f"Evoked not found: {f}")
    ev = mne.read_evokeds(str(f), verbose="ERROR")[0]
    return ev


def roi_trace_mean(ev: mne.Evoked, roi_names_csv: str) -> np.ndarray:
    roi_chs = [c.strip() for c in str(roi_names_csv).split(",") if c.strip()]
    # Solo los canales presentes
    present = [c for c in roi_chs if c in ev.ch_names]
    if len(present) == 0:
        return np.array([], dtype=float)
    data = ev.copy().pick(present).data  # shape (n_ch, n_times)
    return data.mean(axis=0)


def crop_window(ev: mne.Evoked, trace: np.ndarray, tmin: float, tmax: float) -> np.ndarray:
    times = ev.times
    idx = np.where((times >= tmin) & (times <= tmax))[0]
    if idx.size == 0:
        return np.array([], dtype=float)
    return trace[idx]


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    if x.size == 0:
        return x
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if not np.isfinite(sd) or sd < 1e-12:
        return x - mu
    return (x - mu) / sd


def main():
    if not DDS_ALL.exists():
        raise FileNotFoundError(f"Missing: {DDS_ALL}")

    df = pd.read_csv(DDS_ALL)

    # Nos quedamos con los fits válidos por defecto (puedes cambiarlo luego)
    # Aun así guardamos la info de fit_issues para trazabilidad.
    df_use = df.copy()

    rows_out = []
    n = len(df_use)

    for i, row in df_use.iterrows():
        try:
            ev = load_evoked_for_row(row)
            tr = roi_trace_mean(ev, row["roi_names"])
            seg = crop_window(ev, tr, float(row["tmin"]), float(row["tmax"]))
            seg = zscore(seg)

            bins = freedman_diaconis_bins(seg)
            h = shannon_entropy_hist(seg, bins=bins)

            rows_out.append({
                "subject": int(row["subject"]),
                "site": row["site"],
                "cond": row["cond"],
                "window": row["window"],
                "tmin": float(row["tmin"]),
                "tmax": float(row["tmax"]),
                "roi_names": row["roi_names"],
                "agg_used_for_entropy": "mean",
                "entropy_bins": int(bins),
                "entropy_shannon_bits": h,
                "fit_valid": bool(row.get("fit_valid", True)),
                "fit_issues": row.get("fit_issues", np.nan),
                "file": row["file"],
            })

        except Exception as e:
            rows_out.append({
                "subject": int(row["subject"]),
                "site": row.get("site", "NA"),
                "cond": row.get("cond", "NA"),
                "window": row.get("window", "NA"),
                "tmin": row.get("tmin", np.nan),
                "tmax": row.get("tmax", np.nan),
                "roi_names": row.get("roi_names", np.nan),
                "agg_used_for_entropy": "mean",
                "entropy_bins": np.nan,
                "entropy_shannon_bits": np.nan,
                "fit_valid": row.get("fit_valid", np.nan),
                "fit_issues": row.get("fit_issues", np.nan),
                "file": row.get("file", np.nan),
                "error": repr(e),
            })

        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"[{i+1}/{n}] processed")

    out_df = pd.DataFrame(rows_out)
    OUT_ENT.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_ENT, index=False)
    print(f"[OK] wrote: {OUT_ENT} rows={len(out_df)}")

    # Merge con DDS params para análisis estadístico
    key = ["subject", "site", "cond", "window", "file"]
    merged = df.merge(out_df[key + ["entropy_shannon_bits", "entropy_bins", "agg_used_for_entropy"]],
                      on=key, how="left")

    merged.to_csv(OUT_MERGED, index=False)
    print(f"[OK] wrote: {OUT_MERGED} rows={len(merged)} cols={len(merged.columns)}")

    # Resumen rápido
    ok = merged["entropy_shannon_bits"].notna().mean()
    print(f"[OK] entropy availability: {ok*100:.1f}%")
    print(merged.groupby(["window"])["entropy_shannon_bits"].describe())


if __name__ == "__main__":
    main()

