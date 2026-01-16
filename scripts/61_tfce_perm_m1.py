# scripts/46_tfce_perm_mi_active_minus_sham_m1.py
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from mne.stats import permutation_cluster_1samp_test

import matplotlib.pyplot as plt

INP_MI = Path("outputs/features/info_mutual_information_channelwise.csv")
EVOKED_EXAMPLE = Path("outputs/exports/evokeds/sub-01/sub-01_M1_active_ave.fif")

OUT_DIR = Path("outputs/stats/cluster_mi_tfce")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_SUMMARY = OUT_DIR / "tfce_m1_summary.txt"
OUT_CSV     = OUT_DIR / "tfce_m1_stats.csv"
OUT_FIG     = OUT_DIR / "tfce_m1_topomap.png"

N_PERM = 5000
# TFCE one-sided negativo: step debe ser < 0 cuando tail == -1
TFCE_THRESHOLD = dict(start=0.0, step=-0.2)


def main():
    print("=== TFCE permutation: MI (ACTIVE - SHAM), site=M1 ===")
    df = pd.read_csv(INP_MI)
    df = df[(df["reason"] == "ok") & (df["site"] == "m1")].copy()

    piv = df.pivot_table(
        index=["subject", "channel"],
        columns="cond",
        values="mi_early_late_bits",
        aggfunc="mean"
    ).reset_index()

    piv = piv.dropna(subset=["active", "sham"]).copy()
    piv["diff"] = piv["active"] - piv["sham"]

    mat = piv.pivot(index="subject", columns="channel", values="diff")
    mat = mat.dropna(axis=0, how="any")

    channels = mat.columns.to_numpy()
    X = mat.to_numpy()

    ev = mne.read_evokeds(str(EVOKED_EXAMPLE), verbose="ERROR")[0]
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        ev.set_montage(montage, match_case=False, on_missing="ignore")
    except Exception as e:
        print("[WARN] montage:", repr(e))

    # asegurar canales en info
    picks = mne.pick_types(ev.info, eeg=True, meg=False)
    eeg_names = [ev.ch_names[i] for i in picks]
    channels_in_info = [ch for ch in channels if ch in eeg_names]
    if len(channels_in_info) != len(channels):
        missing = sorted(set(channels) - set(channels_in_info))
        print("[WARN] channels not found in evoked info (excluded):", missing)

    keep_idx = [list(channels).index(ch) for ch in channels_in_info]
    X = X[:, keep_idx]
    channels = np.array(channels_in_info)

    info_sub = ev.copy().pick(channels.tolist()).info
    adjacency, ch_names = mne.channels.find_ch_adjacency(info_sub, ch_type="eeg")
    assert list(ch_names) == list(channels), "Channel order mismatch between adjacency and data."

    # One-sided negativo: esperamos ACTIVE-SHAM < 0
    T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        X,
        n_permutations=N_PERM,
        threshold=TFCE_THRESHOLD,   # <- TFCE
        tail=-1,
        adjacency=adjacency,
        out_type="mask",
        seed=7,
        verbose=True
    )

    # Guardar stats por canal (para inspección)
    out = pd.DataFrame({"channel": channels, "T_obs": T_obs})
    out.to_csv(OUT_CSV, index=False)

    # Identificar cluster ganador (menor p)
    best_id = None
    best_p = np.nan
    best_mask = None
    best_channels = []

    if cluster_pv is not None and len(cluster_pv) > 0:
        best_id = int(np.argmin(cluster_pv))
        best_p = float(cluster_pv[best_id])
        best_mask = np.asarray(clusters[best_id], dtype=bool)
        best_channels = channels[np.where(best_mask)[0]].tolist()

    n_cl = int(len(cluster_pv)) if cluster_pv is not None else 0

    # Summary
    lines = []
    lines.append(f"n_subjects={X.shape[0]} n_channels={X.shape[1]}")
    lines.append(f"n_permutations={N_PERM} tfce={TFCE_THRESHOLD} tail=-1")
    lines.append(f"clusters_total={n_cl}")
    lines.append(f"best_cluster_id={best_id}")
    lines.append(f"best_cluster_p={best_p}")
    lines.append(f"best_cluster_n_channels={len(best_channels)}")
    lines.append("best_cluster_channels=" + (",".join(best_channels) if best_channels else ""))
    lines.append(f"max_abs_T_obs={float(np.nanmax(np.abs(T_obs))):.4f}")

    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")

    print("[OK] wrote:", OUT_SUMMARY)
    print("[OK] wrote:", OUT_CSV)
    print("\n".join(lines))

    # Topomap con máscara del cluster ganador
    if best_mask is not None and best_mask.size == len(channels):
        # EvokedArray: 1 "timepoint" con el score por canal
        data = np.asarray(T_obs, float).reshape(-1, 1)
        ev_topo = mne.EvokedArray(data, info_sub, tmin=0.0)

        fig = ev_topo.plot_topomap(
            times=[0.0],
            scalings=1,
            time_format=f"TFCE best p={best_p:.4g}",
            mask=best_mask.reshape(-1, 1),
            show=False
        )
        fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("[OK] wrote:", OUT_FIG)
    else:
        print("[INFO] No best cluster mask available; topomap not written.")


if __name__ == "__main__":
    main()

