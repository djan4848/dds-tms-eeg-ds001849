
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Title:     run_dds_ds001849.py
Purpose:   Fit Dual Damped Sine (DDS) models to TMS–EEG data from OpenNeuro ds001849.
           Reproduces TEPs for M1, DLPFC, and PPC (active vs sham), aligns with cosine similarity framework.
           Outputs parameter estimates (A1, A2, f1, f2, γ1, γ2) and fits per subject × site × condition.

Author:    Damian Jan (damian.jan@dejsl.com)
Affiliation:
           - Instituto Universitario de Neurociencias, Universidad de La Laguna, Spain
           - Dynamic Enterprise Junction Consulting S.L., Spain

Dataset:   OpenNeuro ds001849 (Freedberg et al., 2020)
Model:     Dual Damped Sine (DDS), as described in [JNM Manuscript, 2025]

Dependencies:
    - Python ≥3.8
    - mne ≥1.2
    - mne-bids ≥0.10
    - numpy, pandas, scipy

License:   MIT License 

Usage:
    python run_dds_ds001849.py

Outputs:
    derivatives/dds_ds001849/
        ├── sub-*/{basename}_times.npy
        ├── sub-*/{basename}_roi.npy
        ├── sub-*/{basename}_dds.npy
        └── dds_params_group.csv

Citation:
    Please cite the corresponding Journal of Neuroscience Methods article if you use this code.

Note:
    Ensure `dds_model.py` is present in the same directory with required functions: `fit_dds`, `roi_average`.
"""

"""
DDS runner for OpenNeuro ds001849 (TMS–EEG, multiple stimulation sites, active vs sham).

- Detecta automáticamente todos los EEG (.vhdr/.edf/.bdf/.set)
- Preprocesa: notch 50/60, 1–80 Hz, inpainting del artefacto del pulso (−2…+8 ms),
  baseline (−50…−5 ms), promedio TEP
- Ajusta DDS (2 componentes) sobre un ROI (por sitio si se puede, o media EEG)
- Extrae etiquetas heurísticas de sitio (m1/dlpfc/ppc) y condición (active/sham)
- Guarda: derivatives/dds_ds001849/dds_params_group.csv + NPys de tiempos/ROI/modelo

Requiere: mne, mne-bids, numpy, pandas  (+ tu dds_model.py en el mismo directorio)
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path

def _lazy_imports():
    global mne, BIDSPath, read_raw_bids
    import mne
    from mne_bids import BIDSPath, read_raw_bids

from dds_model import fit_dds, roi_average

# ----------------- AJUSTES -----------------
BIDS_ROOT = Path("/media/neuraldyn/Extreme SSD/GABACOG/ds001849")  # <-- CAMBIA ESTA RUTA
DERIV_ROOT = BIDS_ROOT / "derivatives" / "dds_ds001849"
DERIV_ROOT.mkdir(parents=True, exist_ok=True)

PEAK2PEAK_REJECT = None
FIT_TMIN = 0.005
FIT_TMAX = 0.100
ART_WIN = (-0.002, 0.008)

# ROI por sitio (fallback: media de EEG)
ROI_DEFAULT = ["FC4","C4","CP4"]
ROI_MAP = {
    "m1":    ["FC4","C4","CP4","C2","CP2"],
    "dlpfc": ["AF4","F4","F2","F6","FC4"],
    "ppc":   ["P2","P4","P6","CP4","PO4"],
}

_VALID_EEG_EXT = {".vhdr",".edf",".bdf",".set"}
# -------------------------------------------

def parse_site_cond_from_name(name: str):
    s = name.lower()
    site = None
    for key in ("m1","motor","dlpfc","pfc","frontal","ppc","parietal"):
        if key in s:
            if key in ("m1","motor"):
                site = "m1"
            elif key in ("dlpfc","pfc","frontal"):
                site = "dlpfc"
            else:
                site = "ppc"
            break
    cond = "sham" if ("sham" in s or "placebo" in s) else ("active" if any(k in s for k in ("real","active","tms")) else None)
    return site, cond

def enumerate_paths():
    _lazy_imports()
    eeg_paths = []
    # Lista de sujetos a partir de la estructura BIDS
    bp = BIDSPath(root=BIDS_ROOT, datatype="eeg", suffix="eeg")
    for p in bp.match():
        if p.extension in _VALID_EEG_EXT and p.subject is not None:
            eeg_paths.append(p)
    # deduplicar
    uniq = {}
    for p in eeg_paths:
        key = (p.subject, p.session, p.task, p.acquisition, p.run, p.extension, p.basename)
        uniq[key] = p
    return list(uniq.values())

def detect_pulse_onsets(raw):
    _lazy_imports()
    onsets = []
    if raw.annotations is not None and len(raw.annotations):
        for a in raw.annotations:
            desc = str(a["description"]).upper()
            if ("TMS" in desc) or ("STIMULUS" in desc) or ("S " in desc) or ("TRIG" in desc):
                onsets.append(a["onset"])
    events, _ = mne.events_from_annotations(raw, verbose=False)
    if events.size > 0:
        onsets.extend(raw.times[events[:,0]])
    if not onsets:
        try:
            ev = mne.find_events(raw, shortest_event=1)
            if ev.size > 0:
                onsets.extend(raw.times[ev[:,0]])
        except Exception:
            pass
    return sorted(onsets)

def inpaint_linear(raw, onsets, art_win):
    _lazy_imports()
    raw = raw.copy()
    sfreq = raw.info["sfreq"]
    picks = mne.pick_types(raw.info, eeg=True)
    data = raw.get_data(picks=picks)
    for t0 in onsets:
        i0 = int((t0 + art_win[0]) * sfreq)
        i1 = int((t0 + art_win[1]) * sfreq)
        i0 = max(i0, 1)
        i1 = min(i1, data.shape[1]-2)
        if i1 <= i0:
            continue
        left = data[:, i0-1]
        right = data[:, i1+1]
        for ch in range(data.shape[0]):
            data[ch, i0:i1+1] = np.linspace(left[ch], right[ch], i1 - i0 + 1)
    raw._data[picks, :] = data
    return raw

def preprocess_and_epoch(bids_path, h_freq=100.0, notch_base=60.0):
    _lazy_imports()
    raw = read_raw_bids(bids_path, verbose="ERROR")
    raw.load_data()

    # 1) Seleccionar por TIPO de canal (eeg/eog/stim/misc), sin MEG
    picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=True,
                           stim=True, misc=True, exclude=[])
    raw.pick(picks)

    # 2) Detectar y "inpaint" del pulso ANTES de filtrar (evita ringing)
    onsets = detect_pulse_onsets(raw)
    if onsets:
        raw = inpaint_linear(raw, onsets, ART_WIN)

    # 3) Notch: 60 Hz y (opcional) 120 Hz si el low-pass es alto
    freqs = [notch_base] + ([2 * notch_base] if h_freq >= 110.0 else [])
    raw.notch_filter(freqs=freqs, method='fir', phase='zero')

    # 4) Band-pass recomendado (1–100 Hz por defecto; prueba 120 si evalúas sensibilidad)
    raw.filter(l_freq=1.0, h_freq=h_freq, method='fir', phase='zero', fir_design='firwin')

    # 5) Eventos y epocado
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    if events.size == 0 or not event_id:
        raise RuntimeError("No events found")

    epochs = mne.Epochs(
        raw, events, event_id=list(event_id.values()),
        tmin=-0.100, tmax=0.200, baseline=(-0.050, -0.005),
        preload=True, reject=PEAK2PEAK_REJECT, detrend=0,
        event_repeated="merge"
    )

    # 6) Referencia promedio
    epochs = epochs.set_eeg_reference("average", projection=False)
    return epochs


def run_all():
    _lazy_imports()
    rows = []
    paths = enumerate_paths()
    if not paths:
        print("No EEG files found under", BIDS_ROOT)
        return
    for p in paths:
        try:
            epochs = preprocess_and_epoch(p)
        except Exception as e:
            print(f"[{p.basename}] skipped: {e}")
            continue
        evk = epochs.average()
        site, cond = parse_site_cond_from_name(p.basename)
        # ROI por sitio si existe, si no ROI_DEFAULT, si no media EEG
        if site and site in ROI_MAP:
            picks = [ch for ch in ROI_MAP[site] if ch in evk.ch_names]
        else:
            picks = [ch for ch in ROI_DEFAULT if ch in evk.ch_names]
        y = roi_average(evk, picks) if picks else evk.copy().pick("eeg").data.mean(axis=0)
        t = evk.times
        try:
            params, y_hat = fit_dds(time_s, roi_1d,
                        tmin=0.005, tmax=0.100,
                        two_components=True, t0=0.0,
                        h_freq=100.0)   # <-- coherente con tu filtro
        except Exception as e:
            print(f"[{p.basename}] DDS fit failed: {e}")
            continue
        params.update({
            "subject": p.subject, "session": p.session, "task": p.task, "acq": p.acquisition,
            "run": p.run, "extension": p.extension, "basename": p.basename,
            "site_guess": site, "cond_guess": cond
        })
        out_dir = DERIV_ROOT / f"sub-{p.subject}"
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"{p.basename}_times.npy", t)
        np.save(out_dir / f"{p.basename}_roi.npy", y)
        np.save(out_dir / f"{p.basename}_dds.npy", yhat)
        rows.append(params)
    if not rows:
        print("No results produced.")
        return
    df = pd.DataFrame(rows)
    df.to_csv(DERIV_ROOT / "dds_params_group.csv", index=False)
    print("Saved:", DERIV_ROOT / "dds_params_group.csv")

if __name__ == "__main__":
    run_all()
