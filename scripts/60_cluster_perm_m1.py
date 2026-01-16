from pathlib import Path
import numpy as np
import pandas as pd

import mne
from mne.stats import permutation_cluster_1samp_test

import matplotlib.pyplot as plt

INP_MI = Path("outputs/features/info_mutual_information_channelwise.csv")
EVOKED_EXAMPLE = Path("outputs/exports/evokeds/sub-01/sub-01_M1_active_ave.fif")

OUT_DIR = Path("outputs/stats/cluster_mi")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_CLUSTERS = OUT_DIR / "cluster_perm_m1_active_minus_sham.csv"
OUT_SUMMARY  = OUT_DIR / "cluster_perm_m1_summary.txt"
OUT_FIG      = OUT_DIR / "cluster_perm_m1_topomap.png"

N_PERM = 5000
ALPHA = 0.05
TWO_SIDED = False

def main():
    print("=== Cluster permutation: MI (ACTIVE - SHAM) channel-wise, site=M1 ===")
    print("Input MI:", INP_MI)
    print("Evoked example:", EVOKED_EXAMPLE)
    print("Permutations:", N_PERM)

    df = pd.read_csv(INP_MI)
    df = df[df["reason"] == "ok"].copy()
    df = df[df["site"] == "m1"].copy()
    if df.empty:
        raise RuntimeError("No rows for site=m1 in MI channelwise table.")

    piv = df.pivot_table(index=["subject", "channel"], columns="cond",
                         values="mi_early_late_bits", aggfunc="mean").reset_index()
    if not {"active", "sham"} <= set(piv.columns):
        raise RuntimeError("Missing active/sham columns after pivot. Check input file.")
    piv = piv.dropna(subset=["active", "sham"]).copy()
    piv["diff"] = piv["active"] - piv["sham"]

    subjects = np.sort(piv["subject"].unique())
    channels = np.sort(piv["channel"].unique())
    mat = piv.pivot(index="subject", columns="channel", values="diff").reindex(index=subjects, columns=channels)

    print("[DBG] subjects:", len(subjects), "channels:", len(channels))
    if mat.isna().any().any():
        na_counts = mat.isna().sum(axis=1)
        bad_subs = na_counts[na_counts > 0].index.tolist()
        print("[WARN] subjects with missing channels:", bad_subs)
        mat = mat.dropna(axis=0, how="any")
        subjects = mat.index.to_numpy()
        print("[DBG] after dropna subjects:", len(subjects))

    X = mat.to_numpy()
    if X.shape[0] < 8:
        raise RuntimeError(f"Too few subjects after cleaning: {X.shape[0]}")

    ev = mne.read_evokeds(str(EVOKED_EXAMPLE), verbose="ERROR")[0]
    try:
        montage = mne.channels.make_standard_montage("standard_1020")
        ev.set_montage(montage, match_case=False, on_missing="ignore")
    except Exception as e:
        print("[WARN] could not set montage:", repr(e))

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

    tail = 0 if TWO_SIDED else -1
    T_obs, clusters, cluster_pv, H0 = permutation_cluster_1samp_test(
        X,
        n_permutations=N_PERM,
        threshold=None,   # usa t-threshold automático
        tail=tail,
        adjacency=adjacency,
        out_type="mask",
        verbose=True,
        seed=7
    )

    # --- construir tabla de clusters (puede ser vacía) ---
    rows = []
    if clusters is not None and len(clusters) > 0:
        for i, (mask, p) in enumerate(zip(clusters, cluster_pv)):
            idx = np.where(mask)[0]
            chs = ",".join(channels[idx].tolist())
            rows.append({
                "cluster_id": i,
                "p_value": float(p),
                "n_channels": int(idx.size),
                "channels": chs,
                "T_sum": float(T_obs[idx].sum()),
                "T_mean": float(T_obs[idx].mean()) if idx.size else np.nan,
            })

    out = pd.DataFrame(rows, columns=["cluster_id","p_value","n_channels","channels","T_sum","T_mean"])
    out = out.sort_values("p_value") if len(out) else out
    out.to_csv(OUT_CLUSTERS, index=False)

    # --- resumen robusto incluso sin clusters ---
    lines = []
    lines.append(f"n_subjects={X.shape[0]} n_channels={X.shape[1]}")
    lines.append(f"n_permutations={N_PERM} two_sided={TWO_SIDED}")
    lines.append(f"max|T_obs|={float(np.nanmax(np.abs(T_obs))):.4f}")
    lines.append(f"clusters_total={len(out)}")
    if len(out):
        sig = out[out["p_value"] <= ALPHA]
        lines.append(f"clusters_sig(p<={ALPHA})={len(sig)}")
        if len(sig):
            lines.append("\nTop significant clusters:")
            lines.append(sig.head(5).to_string(index=False))
        else:
            lines.append("\nNo significant clusters at alpha=" + str(ALPHA))
    else:
        lines.append("\nNo clusters formed at the initial cluster-forming threshold (likely no sensors exceeded |t| threshold).")

    OUT_SUMMARY.write_text("\n".join(lines), encoding="utf-8")

    print("[OK] wrote:", OUT_CLUSTERS)
    print("[OK] wrote:", OUT_SUMMARY)
    print("\n".join(lines))

    # topomap solo si hay clusters
    if len(out):
        best = out.iloc[0]
        best_id = int(best["cluster_id"])
        best_p = float(best["p_value"])

        mask = np.asarray(clusters[best_id], dtype=bool)

        data = T_obs.reshape(-1, 1)
        ev_topo = mne.EvokedArray(data, info_sub, tmin=0.0)
        fig = ev_topo.plot_topomap(
            times=[0.0],
            scalings=1,
            time_format=f"Best cluster p={best_p:.4g}",
            mask=mask.reshape(-1, 1),
            show=False
        )
        fig.savefig(OUT_FIG, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print("[OK] wrote:", OUT_FIG)

if __name__ == "__main__":
    main()
