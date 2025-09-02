#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:     run_cosine_similarity_ds001849.py
Purpose:   Compute cosine similarity of TMS–EEG TEPs from the OpenNeuro ds001849 dataset.
           Reproduces the analysis from Freedberg et al. (2020), using binarized temporal derivatives
           of evoked responses across stimulation sites (M1, DLPFC, PPC) and conditions (active vs sham).

Author:    Damian Jan (damian.jan@dejsl.com)
Affiliation:
           - Instituto Universitario de Neurociencias, Universidad de La Laguna, Spain
           - Dynamic Enterprise Junction Consulting S.L., Spain

Dataset:   OpenNeuro ds001849 (Freedberg et al., 2020)

Dependencies:
    - Python ≥3.8
    - mne ≥1.2
    - mne-bids ≥0.10
    - numpy, pandas, scipy, matplotlib

Main steps:
    1. Load BIDS-organized EEG data and detect TMS pulse artifacts
    2. Linearly inpaint artifacts (−2 to +8 ms)
    3. Apply notch (60 Hz), bandpass (1–100 Hz), and average referencing
    4. Epoch and extract evoked responses per subject × site × condition
    5. Compute cosine similarity between site pairs on binarized derivatives
    6. Save similarity matrix per subject and condition to CSV

Outputs:
    derivatives/cosine_similarity_ds001849/cosine_similarity_results.csv

Usage:
    python run_cosine_similarity_ds001849.py

License:   MIT License 

Citation:
    Please cite:
      - Freedberg et al., PLoS ONE (2020): https://doi.org/10.1371/journal.pone.0216185
      - And the accompanying Journal of Neuroscience Methods article describing this reproduction.

Notes:
    - Compatible with BIDS-structured EEG datasets
    - Assumes filenames encode stimulation site and condition (e.g., 'sub-01_task-dlpfc_sham.vhdr')
"""



import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mne_bids import BIDSPath, read_raw_bids
import mne

# --- CONFIGURATION ---
BIDS_ROOT = Path("/media/neuraldyn/Extreme SSD/GABACOG/ds001849")  # <-- update if needed
DERIV_ROOT = BIDS_ROOT / "derivatives" / "cosine_similarity_ds001849"
DERIV_ROOT.mkdir(parents=True, exist_ok=True)

# TMS artifact window (in seconds)
ART_WIN = (-0.002, 0.008)
# Epoch window
EPOCH_TMIN, EPOCH_TMAX = -0.1, 0.5
# Baseline
BASELINE = (-0.05, -0.005)

# ---- FUNCTIONAL BLOCKS ----
def inpaint_linear(raw, onsets, art_win):
    """Linear inpainting of TMS pulse artifact."""
    raw = raw.copy()
    sfreq = raw.info["sfreq"]
    picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=picks)
    for t0 in onsets:
        i0 = int((t0 + art_win[0]) * sfreq)
        i1 = int((t0 + art_win[1]) * sfreq)
        i0 = max(i0, 1)
        i1 = min(i1, data.shape[1] - 2)
        if i1 <= i0:
            continue
        left = data[:, i0 - 1]
        right = data[:, i1 + 1]
        for ch in range(data.shape[0]):
            data[ch, i0:i1 + 1] = np.linspace(left[ch], right[ch], i1 - i0 + 1)
    raw._data[picks, :] = data
    return raw

def binarized_derivative(signal):
    """Return the sign of the temporal derivative of the signal."""
    return np.sign(np.diff(signal, axis=-1))

def cosine_similarity(a, b):
    """Cosine similarity between two 2D matrices (channels × time)."""
    a = a.reshape(a.shape[0], -1)
    b = b.reshape(b.shape[0], -1)
    dot = np.sum(a * b, axis=1)
    norm_a = np.linalg.norm(a, axis=1)
    norm_b = np.linalg.norm(b, axis=1)
    sim = dot / (norm_a * norm_b + 1e-8)
    return np.nanmean(sim)

def extract_evoked(raw):
    """Preprocessing and epoching."""
    raw.load_data()
    raw.pick_types(eeg=True)
    onsets = [ann["onset"] for ann in raw.annotations]
    raw = inpaint_linear(raw, onsets, ART_WIN)
    raw.notch_filter(freqs=[60.0], method="fir")
    raw.filter(1., 100., method="fir")

    events, event_id = mne.events_from_annotations(raw)
    if len(event_id) == 0:
        raise RuntimeError("No event markers found")

    epochs = mne.Epochs(raw, events, event_id=list(event_id.values()), 
                        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
                        baseline=BASELINE, detrend=0, preload=True)
    epochs.set_eeg_reference("average", projection=False)
    return epochs.average()

def label_condition(path_str):
    name = path_str.lower()
    if "sham" in name:
        return "sham"
    if "active" in name or "real" in name or "tms" in name:
        return "active"
    return "unknown"

def label_site(path_str):
    name = path_str.lower()
    if "motor" in name or "m1" in name:
        return "m1"
    if "frontal" in name or "dlpfc" in name:
        return "dlpfc"
    if "parietal" in name or "ppc" in name:
        return "ppc"
    return "unknown"

# ---- MAIN ANALYSIS ----
all_paths = list(BIDSPath(root=BIDS_ROOT, datatype="eeg", suffix="eeg").match())
results = []

for p in all_paths:
    try:
        raw = read_raw_bids(p, verbose="ERROR")
        evoked = extract_evoked(raw)
        cond = label_condition(p.basename)
        site = label_site(p.basename)
        subj = p.subject
        deriv = binarized_derivative(evoked.data)
        results.append({
            "subject": subj,
            "site": site,
            "condition": cond,
            "deriv": deriv,
            "evoked": evoked
        })
    except Exception as e:
        print(f"[{p.basename}] skipped: {e}")
        continue

# ---- SIMILARITY COMPUTATIONS ----
rows = []
for subj in sorted(set(r["subject"] for r in results)):
    subj_res = [r for r in results if r["subject"] == subj]
    for cond in ("active", "sham"):
        by_cond = {r["site"]: r for r in subj_res if r["condition"] == cond}
        if len(by_cond) >= 2:
            sites = list(by_cond.keys())
            for i in range(len(sites)):
                for j in range(i+1, len(sites)):
                    A = by_cond[sites[i]]["deriv"]
                    B = by_cond[sites[j]]["deriv"]
                    sim = cosine_similarity(A, B)
                    rows.append({
                        "subject": subj,
                        "cond": cond,
                        "site1": sites[i],
                        "site2": sites[j],
                        "similarity": sim
                    })

df = pd.DataFrame(rows)
df.to_csv(DERIV_ROOT / "cosine_similarity_results.csv", index=False)
print("Saved:", DERIV_ROOT / "cosine_similarity_results.csv")

