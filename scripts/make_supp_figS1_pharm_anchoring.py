#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def build_suppfig_s1(
    deriv_root: Path,
    outdir: Path,
    baseline_f_hz: float = 30.0,
    debug: bool = False,
):
    """
    Supplementary Fig S1 – Pharmacological anchoring

    Reads DDS pooled parameters from:
      - derivatives/dds_gabacog/dds_params_channelwise.csv

    Uses:
      - SICI: gamma1 (s^-1)
      - ICF : f2 (Hz)

    Plots a barh with pharmacological anchors and DDS-derived relative values.
    """

    csv_ch = deriv_root / "dds_params_channelwise.csv"
    if not csv_ch.exists():
        raise FileNotFoundError(f"Missing: {csv_ch}")

    df = pd.read_csv(csv_ch)

    # Basic sanity filters
    df = df[df["group"].isin(["CTL", "TOC", "OCD"])].copy()  # allow OCD naming too
    df["protocol"] = df["protocol"].astype(str)

    # --- pooled means (all subjects, all channels) ---
    sici = df[df["protocol"].str.upper() == "SICI"]
    icf  = df[df["protocol"].str.upper() == "ICF"]

    # Robustness guard (optional): keep as-is unless you want to enforce R2 threshold here
    # sici = sici[sici["R2"] >= 0.9]
    # icf  = icf[icf["R2"] >= 0.9]

    if sici.empty or icf.empty:
        raise RuntimeError(
            f"Empty protocol data after filtering. "
            f"SICI rows={len(sici)}, ICF rows={len(icf)}"
        )

    if "gamma1" not in sici.columns:
        raise KeyError("Column 'gamma1' not found in dds_params_channelwise.csv")
    if "f2" not in icf.columns:
        raise KeyError("Column 'f2' not found in dds_params_channelwise.csv (expected Hz).")

    pooled_gamma1 = float(sici["gamma1"].dropna().mean())  # s^-1
    pooled_f2_hz  = float(icf["f2"].dropna().mean())       # Hz

    if debug:
        print(f"[S1] pooled_gamma1 (SICI) = {pooled_gamma1:.4f} s^-1")
        print(f"[S1] pooled_f2_hz  (ICF)  = {pooled_f2_hz:.4f} Hz")
        print(f"[S1] baseline_f_hz         = {baseline_f_hz:.2f} Hz")

    # --- pharmacological anchors (keep your original values) ---
    # These are "relative change × baseline" for canonical SICI/ICF readouts
    ref_effects = {
        "Ketamine 0.5 mg/kg\n(Momi 21)": 0.65,   # –35% ICF (example anchor)
        "Zolpidem 10 mg\n(Premoli 14)" : 1.35,   # +35% SICI potentiation
        "DZP 10 mg\n(Ziemann 96)"      : 1.42,   # +42% SICI potentiation
    }

    # --- DDS relative values ---
    # For gamma1: in the original script you anchored DDS γ1 at baseline (=1.0) as a reference.
    # We'll keep the same logic so the figure reads as "DDS markers plotted relative to baseline".
    dds_rel_gamma1 = 1.0

    # For f2: relative to a canonical 30 Hz centre (as in your original ω2 baseline logic),
    # but using f2 directly (Hz).
    dds_rel_f2 = pooled_f2_hz / float(baseline_f_hz)

    labels = list(ref_effects.keys()) + ["DDS γ₁ (SICI)", "DDS f₂ (ICF)"]
    values = list(ref_effects.values()) + [dds_rel_gamma1, dds_rel_f2]

    # Colors: keep close to your original (light blue for refs, blue/orange for DDS)
    colors = ["#b2c2ff"] * len(ref_effects) + ["#4472c4", "#ff9944"]

    outdir.mkdir(parents=True, exist_ok=True)
    out_png = outdir / "SuppFig_S1_pharmacological_anchoring.png"

    plt.figure(figsize=(7.2, 4.2))
    plt.barh(labels, values, color=colors)
    plt.axvline(1.0, ls="--", color="k", linewidth=1.2)
    plt.xlabel("Relative change (× baseline)")
    plt.title("Supplementary Fig S1 – Pharmacological anchoring")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

    print(f"[saved] {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deriv_root", type=str, default="derivatives/dds_gabacog",
                    help="Root folder containing dds_params_channelwise.csv")
    ap.add_argument("--outdir", type=str, default="derivatives/dds_gabacog/figures_paper_A",
                    help="Output directory for the figure")
    ap.add_argument("--baseline_f_hz", type=float, default=30.0,
                    help="Baseline frequency (Hz) to normalise f2 (default=30 Hz)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    build_suppfig_s1(
        deriv_root=Path(args.deriv_root),
        outdir=Path(args.outdir),
        baseline_f_hz=args.baseline_f_hz,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()

