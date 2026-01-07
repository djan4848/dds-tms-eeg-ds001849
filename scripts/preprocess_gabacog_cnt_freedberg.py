#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
import pandas as pd
import mne
import numpy as np

# ----------------------------
# Helpers
# ----------------------------
from pathlib import Path
import csv
from datetime import datetime

AUDIT_FIELDS = [
    "timestamp",
    "subject",
    "protocol",
    "cnt_file",
    "n_channels_raw",
    "n_channels_final",
    "sfreq_raw",
    "sfreq_final",
    "n_epochs",
    "event_id_used",
    "n_events_detected",
    "bad_channels_n",
    "bad_channels_list",
    "status",
    "notes",
]

def init_audit_csv(audit_path: Path):
    if not audit_path.exists():
        with open(audit_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
            writer.writeheader()

def append_audit_row(audit_path: Path, row: dict):
    with open(audit_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDS)
        writer.writerow(row)

SUBJECT_RE = re.compile(r"sub_(\d{3})")#SUBJECT_RE = re.compile(r"^(\d{3})_")

def parse_subject_id_from_fname(fname: str) -> str | None:
    base = Path(fname).name
    m = SUBJECT_RE.match(base)
    return m.group(1) if m else None

def load_cnt_list(csv_path: Path) -> list[str]:
    # Your people_*.csv is effectively a 1-column list (sometimes header-like first row)
    df = pd.read_csv(csv_path, header=None)
    items = [str(x).strip() for x in df.iloc[:, 0].tolist()
             if str(x).strip() and str(x).strip().lower() != "nan"]
    return items
def force_eeg_channel_types(raw: mne.io.BaseRaw, debug: bool = False) -> mne.io.BaseRaw:
    """
    Ensure CNT channels are typed as EEG so ICA and montage work.
    Many CNT imports come as 'misc' unless explicitly set.
    """
    # Heuristic: anything that's not clearly stim/eog/ecg/emg becomes EEG.
    # If your CNT contains a trigger/stim channel name, add it to this list.
    stim_like = {"STI 014", "STI014", "TRIGGER", "TRIG", "STATUS", "Stim", "STIM"}
    eog_like  = {"EOG", "HEOG", "VEOG"}

    mapping = {}
    for ch in raw.ch_names:
        ch_up = ch.upper()
        if ch in stim_like or any(s in ch_up for s in stim_like):
            mapping[ch] = "stim"
        elif any(s in ch_up for s in eog_like):
            mapping[ch] = "eog"
        else:
            mapping[ch] = "eeg"

    raw.set_channel_types(mapping, verbose="ERROR")

    if debug:
        types = raw.get_channel_types(unique=True)
        print(f"[DEBUG] Channel types after forcing: {types}")

    return raw

def normalize_channel_names(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    new = {}
    #for ch in raw.ch_names:
    #    c = ch.strip()
    #    c = c.replace("EEG ", "").replace("EEG_", "").replace("EEG-", "")
    #    c = c.replace("-REF", "").replace("_REF", "")
    #    c = c.replace(".", "").strip()

    #    if len(c) >= 2:
   #         c2 = c[0].upper() + c[1:].lower()
    #    else:
   #         c2 = c.upper()

    #    c2 = c2.replace("Z", "z")  # FZ->Fz etc.
    #    new[ch] = c2

    #raw.rename_channels(new, verbose="ERROR")
    rename_dict = {
                    'FP1': 'Fp1',
                    'FP2': 'Fp2',
                    'FZ': 'Fz',
                    'CZ': 'Cz',
                    'PZ': 'Pz',
                    'M1': 'TP9',
                    'M2': 'TP10',
                    'FPZ': 'Fpz',
                    'CPZ': 'CPz',
                    'POZ': 'POz',
                    'OZ': 'Oz',
                  
                }
    raw.rename_channels(rename_dict)

    channels_to_remove=['Iz', 'BP1', 'BP2','HEO', 'VEO', 'EKG', 'EMG'] #no están conectados

    #Remove channels listed in channels_to_remove
    #channels_to_remove=['Iz','BP1', 'BP2', 'BP3','BP4','-1','-0']
    raw.drop_channels(channels_to_remove)
    return raw

def anonymize_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    # Remove potentially identifying info
    try:
        raw.set_meas_date(None)
    except Exception:
        pass
    try:
        raw.info["subject_info"] = None
    except Exception:
        pass
    try:
        if raw.annotations is not None:
            raw.annotations.orig_time = None
    except Exception:
        pass
    return raw

def crop_to_event_span_by_key(raw: mne.io.BaseRaw, event_key: str) -> mne.io.BaseRaw:
    """
    Crop raw to [first_event, last_event] using the annotation key (e.g. '32').
    """
    events, event_dict = mne.events_from_annotations(raw, verbose="ERROR")
    if len(events) == 0:
        return raw
    if event_key not in event_dict:
        return raw

    target_id = event_dict[event_key]
    ev = events[events[:, 2] == target_id]
    if len(ev) == 0:
        return raw

    sfreq = float(raw.info["sfreq"])
    t0 = float(ev[:, 0].min()) / sfreq
    t1 = float(ev[:, 0].max()) / sfreq
    t0 = max(t0, 0.0)
    t1 = min(t1, float(raw.times[-1]))
    if t1 > t0:
        raw = raw.crop(tmin=t0, tmax=t1)
    return raw


# ----------------------------
# Freedberg preprocessing import (robust)
# ----------------------------

def run_freedberg_preproc(raw: mne.io.BaseRaw, line_freq: float = 50.0, freedberg_path: str | None = None) -> mne.io.BaseRaw:
    """
    Load preprocess_freedberg_mne robustly (no package assumptions).
    If freedberg_path is provided, load it from that file.
    Otherwise search scripts/freedberg_mne.py or repo_root/freedberg_mne.py.
    """
    import importlib.util

    def load_from_file(pyfile: Path):
        spec = importlib.util.spec_from_file_location("freedberg_mne", str(pyfile))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load module spec from: {pyfile}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore
        if not hasattr(mod, "preprocess_freedberg_mne"):
            raise ImportError(f"'preprocess_freedberg_mne' not found in: {pyfile}")
        return mod.preprocess_freedberg_mne

    if freedberg_path:
        pyfile = Path(freedberg_path).expanduser().resolve()
        if not pyfile.exists():
            raise FileNotFoundError(f"--freedberg_path not found: {pyfile}")
        preprocess_freedberg_mne = load_from_file(pyfile)
    else:
        here = Path(__file__).resolve().parent
        candidates = [
            here / "freedberg_mne.py",
            here.parent / "freedberg_mne.py",
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            raise ModuleNotFoundError(
                "Cannot locate freedberg_mne.py. Put it in scripts/ or repo root, "
                "or pass --freedberg_path /path/to/freedberg_mne.py"
            )
        preprocess_freedberg_mne = load_from_file(found)

    # IMPORTANT: adjust this keyword if your freedberg_mne uses another name
    clean,_ = preprocess_freedberg_mne(raw=raw,montage='standard_1020', notch=float(line_freq))
    return clean


# ----------------------------
# Main per-protocol processing
# ----------------------------

def preprocess_one_cnt(cnt_path, subject_id, protocol, out_dir, tms_event_key, 
                       recode_to, tmin, tmax, baseline, line_freq, freedberg_path,
                       montage_name="standard_1020", debug=False):
    global audit_path
    print(f"\n[DEBUG-START] Procesando: {cnt_path.name}")
    #raw = mne.io.read_raw_cnt(str(cnt_path), preload=True, verbose="ERROR")
    raw = mne.io.read_raw_fif(str(cnt_path), preload=True, verbose="ERROR")
    audit_row = {
    "timestamp": datetime.now().isoformat(timespec="seconds"),
    "subject": subject_id,
    "protocol": protocol,   # "SICI" o "ICF"
    "cnt_file": cnt_path.name,
    "n_channels_raw": raw.info["nchan"],
    "n_channels_final": None,
    "sfreq_raw": raw.info["sfreq"],
    "sfreq_final": None,
    "n_epochs": None,
    "event_id_used": "32",
    "n_events_detected": None,
    "bad_channels_n": None,
    "bad_channels_list": None,
    "status": "STARTED",
    "notes": "",
    }

    #raw = normalize_channel_names(raw)
    #raw = force_eeg_channel_types(raw, debug=debug)

    # TRAZA 1: Estado inicial tras carga y normalización
    print(f"[DEBUG-TRAZA 1] Canales tras normalizar: {len(raw.ch_names)}")
    print(f"[DEBUG-TRAZA 1] Dig inicial: {'SI' if raw.info['dig'] else 'NO'}")

    try:
        montage = mne.channels.make_standard_montage(montage_name)
        raw.set_montage(montage, on_missing='warn')
        print(f"[DEBUG-TRAZA 2] Montage '{montage_name}' aplicado.")
        print(f"[DEBUG-TRAZA 2] Canales con posición: {len([ch for ch in raw.info['chs'] if not np.all(ch['loc'][:3] == 0)])}")
    except Exception as e:
        print(f"[DEBUG-TRAZA 2] ERROR al aplicar montage: {e}")
        audit_row["status"] = "FAILED"
        audit_row["notes"] = repr(e)
        append_audit_row(audit_path, audit_row)
        raise

        

    # Verificación de eventos
    events, event_dict = mne.events_from_annotations(raw, verbose="ERROR")
    print(f"[DEBUG-TRAZA 3] Eventos detectados: {event_dict}")

    #raw = crop_to_event_span_by_key(raw, event_key=tms_event_key)
    #raw = anonymize_raw(raw)

    # Llamada al proceso principal
    print("[DEBUG-TRAZA 4] Entrando en run_freedberg_preproc...")
    raw_clean = run_freedberg_preproc(raw, line_freq=line_freq, freedberg_path=freedberg_path)
    audit_row["sfreq_final"] = raw_clean.info["sfreq"]

    bads = raw_clean.info.get("bads", [])
    audit_row["bad_channels_n"] = len(bads)
    audit_row["bad_channels_list"] = ",".join(bads)

    # Freedberg-like automatic preprocessing
    #raw_clean = run_freedberg_preproc(raw, line_freq=line_freq, freedberg_path=freedberg_path)

    # Events from cleaned raw
    events, event_dict = mne.events_from_annotations(raw_clean, verbose="ERROR")
    if tms_event_key not in event_dict:
        print(f"[SKIP] {protocol} {subject_id}: TMS key '{tms_event_key}' disappeared after preprocessing. Available: {event_dict}")
        return

    target_id = event_dict[tms_event_key]
    ev = events[events[:, 2] == target_id]
    if len(ev) == 0:
        print(f"[SKIP] {protocol} {subject_id}: no events for key '{tms_event_key}' after selection.")
        return

    # Recode ALL TMS pulses to protocol-specific integer (2 for SICI, 3 for ICF)
    ev = ev.copy()
    ev[:, 2] = int(recode_to)

    # Create epochs with event_id matching your downstream expectations
    event_id = {str(recode_to): int(recode_to)}

    epochs = mne.Epochs(
        raw_clean,
        events=ev,
        event_id=event_id,
        tmin=float(tmin),
        tmax=float(tmax),
        baseline=baseline,
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )

    # Anonymize epochs metadata too
    try:
        epochs.info["subject_info"] = None
    except Exception:
        pass
    try:
        epochs.set_meas_date(None)
    except Exception:
        pass

    out_dir.mkdir(parents=True, exist_ok=True)
    out_fif = out_dir / f"sub_{subject_id}__{protocol}-epo.fif"
    epochs.save(str(out_fif), overwrite=True)
    audit_row["n_channels_final"] = epochs.info["nchan"]
    audit_row["n_epochs"] = len(epochs)
    audit_row["status"] = "OK"

    append_audit_row(audit_path, audit_row)

    print(f"[OK] saved {out_fif} | n_epochs={len(epochs)} | n_chan={len(epochs.ch_names)}")

audit_path = Path("/")
def main():
    global audit_path
    ap = argparse.ArgumentParser()
    ap.add_argument("--home_path", type=str, default="/media/neuraldyn/Extreme SSD/GABACOG/")
    ap.add_argument("--input_sici", type=str, default="procesado_SICI/")
    ap.add_argument("--input_icf", type=str, default="procesado_ICF/")
    ap.add_argument("--output_sici", type=str, default="procesado_SICI/")
    ap.add_argument("--output_icf", type=str, default="procesado_ICF/")

    ap.add_argument("--people_sici_csv", type=str, required=True)
    ap.add_argument("--people_icf_csv", type=str, required=True)

    ap.add_argument("--tms_event_key", type=str, default="32",
                    help="Annotation key for TMS pulses (parallel port): default '32'.")

    ap.add_argument("--tmin", type=float, default=-0.5)
    ap.add_argument("--tmax", type=float, default=0.5)
    ap.add_argument("--baseline_tmin", type=float, default=-0.5)
    ap.add_argument("--baseline_tmax", type=float, default=0.0)

    ap.add_argument("--line_freq", type=float, default=50.0,
                    help="Line noise frequency for notch in Freedberg preproc (Spain=50).")

    ap.add_argument("--freedberg_path", type=str, default="",
                    help="Optional absolute path to freedberg_mne.py if not in scripts/ or repo root.")

    ap.add_argument("--subjects", type=str, default="",
                    help="Optional comma-separated subject ids (e.g. '005,006'). If empty, process all.")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    home = Path(args.home_path)
    in_sici = home / args.input_sici
    in_icf  = home / args.input_icf
    out_sici = home / args.output_sici
    out_icf  = home / args.output_icf

    baseline = (float(args.baseline_tmin), float(args.baseline_tmax))
    freedberg_path = args.freedberg_path.strip() or None

    # Load CNT filename lists
    sici_files = load_cnt_list(Path(args.people_sici_csv))
    icf_files  = load_cnt_list(Path(args.people_icf_csv))
    print('sici_files',sici_files)
    print('icf_files',icf_files)
    # subject -> filename mapping
    map_sici = {parse_subject_id_from_fname(f): f for f in sici_files if parse_subject_id_from_fname(f)}
    map_icf  = {parse_subject_id_from_fname(f): f for f in icf_files  if parse_subject_id_from_fname(f)}
    audit_path = home / 'ds001849'/'repositorio-git'/'derivatives'/'dds_gabacog'/'logs' / "preprocessing_audit.csv"
    init_audit_csv(audit_path)

    # Subjects to process
    if args.subjects.strip():
        wanted = [x.strip() for x in args.subjects.split(",") if x.strip()]
    else:
        wanted = sorted(set(map_sici.keys()) | set(map_icf.keys()))

    print(f"[INFO] Subjects requested: {len(wanted)}")

    # --- SICI: recode TMS '32' -> 2
    for sid in wanted:
        if sid in map_sici:
            cnt_path = in_sici / map_sici[sid]
            if not cnt_path.exists():
                print(f"[SKIP] SICI {sid}: missing {cnt_path}")
                continue
            preprocess_one_cnt(
                cnt_path=cnt_path,
                subject_id=sid,
                protocol="SICI",
                out_dir=out_sici,
                tms_event_key=args.tms_event_key,
                recode_to=2,
                tmin=args.tmin,
                tmax=args.tmax,
                baseline=baseline,
                line_freq=args.line_freq,
                freedberg_path=freedberg_path,
                debug=args.debug,
            )

    # --- ICF: recode TMS '32' -> 3
    for sid in wanted:
        if sid in map_icf:
            cnt_path = in_icf / map_icf[sid]
            if not cnt_path.exists():
                print(f"[SKIP] ICF {sid}: missing {cnt_path}")
                continue
            preprocess_one_cnt(
                cnt_path=cnt_path,
                subject_id=sid,
                protocol="ICF",
                out_dir=out_icf,
                tms_event_key=args.tms_event_key,
                recode_to=3,
                tmin=args.tmin,
                tmax=args.tmax,
                baseline=baseline,
                line_freq=args.line_freq,
                freedberg_path=freedberg_path,
                debug=args.debug,
            )

    print("[DONE] preprocessing complete.")


if __name__ == "__main__":
    main()

