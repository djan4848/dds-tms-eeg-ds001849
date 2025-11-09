# scripts/preprocess_ds001849_from_sets.py
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))  # ver ddstms/

import argparse
from pathlib import Path
import re
import mne
import numpy as np
from mne.channels import make_standard_montage
from mne.preprocessing import compute_current_source_density
from scipy.stats import kurtosis
from scipy.interpolate import CubicSpline
from collections import Counter
import numpy as np
from mne.io import read_raw_eeglab
from ddstms.preproc.freedberg_mne import preprocess_freedberg_mne

SITE_MAP = {"m1": "M1", "dlpfc": "DLPFC", "ppc": "PPC"}
COND_MAP = {"active": "active", "sham": "sham"}

import re

def parse_site_cond_from_name(fname: str):
    """
    Extrae (site, cond) de nombres tipo:
      sub-01_task-tmseegrest_acq-dlpfcactive_eeg.set
      sub-01_task-tmseegrest_acq-m1sham_eeg.set
      sub-01_task-tmseegrest_acq-ppcactive_eeg.set
    Soporta que site y condición estén pegados (sin guión bajo).
    """
    f = fname.lower()
    # patrón: acq-(dlpfc|m1|ppc)(active|sham)
    m = re.search(r'acq-(dlpfc|m1|ppc)(active|sham)\b', f)
    if not m:
        # fallback tolerante: busca cualquier ocurrencia site+cond consecutivas
        sites = ('dlpfc', 'm1', 'ppc')
        conds = ('active', 'sham')
        for s in sites:
            for c in conds:
                if f.find(f'{s}{c}') != -1 and 'acq-' in f:
                    return s.upper(), c
        raise ValueError(f"No pude extraer site/cond de: {fname}")

    site, cond = m.group(1), m.group(2)
    # normaliza site a mayúsculas convencionales
    site = site.upper()  # 'DLPFC', 'M1', 'PPC'
    return site, cond


def _dbg_channel_table(raw):
    rows = []
    for ch in raw.info['chs']:
        name = ch['ch_name']
        ctype = ch['kind'] if isinstance(ch['kind'], str) else raw.get_channel_types(picks=[name])[0]
        loc = np.array(ch['loc'][:3], float)
        rows.append((name, ctype, bool(np.all(np.isfinite(loc))), tuple(loc)))
    rows.sort()
    print("[DBG] Tabla canales (name, type, finite_loc, loc[:3]):")
    for r in rows:
        print("       ", r)

def epochs_from_annotations(raw, tmin=-0.1, tmax=0.5, baseline=(None, 0)):
    # Los eventos de EEGLAB llegan como annotations→events
    events, event_id = mne.events_from_annotations(raw, verbose=False)
    # ====== TRZ 1: Estado de los canales antes de Epochs ======
    print("\n[TRZ] ===== Estado canales ANTES de Epochs =====")
    print("[TRZ] nchan:", raw.info['nchan'])
    print("[TRZ] ch_names:", raw.ch_names)
    print("[TRZ] tipos (Counter):", Counter(raw.get_channel_types()))
    print("[TRZ] bads:", raw.info.get('bads', []))
    print("[TRZ] montage asignado?:", raw.get_montage() is not None)
    _dbg_channel_table(raw)

    picks_excl = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, emg=False,
                                stim=False, misc=False, exclude='bads')
    picks_all = mne.pick_types(raw.info, eeg=True, meg=False, eog=False, ecg=False, emg=False,
                               stim=False, misc=False, exclude=[])
    print("[TRZ] picks_excl (exclude='bads') -> n:", len(picks_excl), " idx:", picks_excl)
    print("[TRZ] picks_all  (exclude=[])      -> n:", len(picks_all),  " idx:", picks_all)
    if len(picks_excl) == 0 and len(picks_all) > 0:
        print("[TRZ] AVISO: todos los EEG están en bads. Eso vacía picks_excl.")
    if len(picks_all) == 0:
        print("[TRZ] ERROR: no se detectan canales EEG en absoluto (tipos mal asignados?).")

    # ===== (tu código actual para construir events) =====
    events, _ = mne.events_from_annotations(raw, event_id=event_id, verbose=False)
    print("[TRZ] eventos:", events.shape[0])

    # ====== TRZ 2: llamada a Epochs con picks calculados ======
    print("[TRZ] Llamando a mne.Epochs(...) con picks_excl")
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=baseline, preload=True, picks=picks_excl, verbose=True)
    print("[TRZ] Epochs OK. shape:", epochs.get_data().shape)
    return epochs
# --- Normalizar nombres a los del standard_1020 (respetando mayúsculas/minúsculas de MNE) ---
_alias_map = {
    # frontales prefrontales
    "FP1": "Fp1", "fp1": "Fp1",
    "FP2": "Fp2", "fp2": "Fp2",
    # línea media (z) típicos
    "FZ": "Fz", "fz": "Fz",
    "CZ": "Cz", "cz": "Cz",
    "PZ": "Pz", "pz": "Pz",
    "OZ": "Oz", "oz": "Oz",
    "IZ": "Iz", "iz": "Iz",
    "AFZ": "AFz", "afz": "AFz",
    "FCZ": "FCz", "fcz": "FCz",
    "CPZ": "CPz", "cpz": "CPz",
    "POZ": "POz", "poz": "POz",
}

def _normalize_name(name: str) -> str:
    s = name.strip()
    # 1) alias directos
    if s in _alias_map:
        return _alias_map[s]
    if s.upper() in _alias_map:
        return _alias_map[s.upper()]
    # 2) FPn/FPnn → Fpn/Fpnn
    m = re.fullmatch(r'fp(\d+)', s.lower())
    if m:
        return f"Fp{m.group(1)}"
    # 3) sufijo Z/z → forzar 'z' final con raíz en mayúscula
    m = re.fullmatch(r'([a-z]+)z', s.lower())
    if m:
        root = m.group(1).upper()
        return f"{root}z"
    # 4) dejar otros tal cual (TP9, TP10, FC6, etc. ya coinciden)
    return s

def _downsample_if_needed(raw, target_sfreq=1000.0):
    sf = float(raw.info['sfreq'])
    if sf > target_sfreq + 1e-6:
        raw.resample(target_sfreq, npad="auto")
    return raw

def _events_from_annotations_strict(raw):
    # Usa todas las anotaciones como posibles pulsos TMS excepto 'boundary'
    anns = raw.annotations
    onsets = anns.onset[anns.description != 'boundary']
    if len(onsets) == 0:
        raise RuntimeError("No se encontraron marcadores TMS en annotations.")
    return (onsets * raw.info['sfreq']).astype(int)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids_root", required=True, help="Ruta al directorio raíz BIDS que contiene todos los sujetos")
    ap.add_argument("--deriv_root", required=True, help="Directorio salida Evokeds limpios")
    args = ap.parse_args()

    bids_root = Path(args.bids_root).resolve()
    deriv_root = Path(args.deriv_root)
    deriv_root.mkdir(parents=True, exist_ok=True)

    # Encuentra todos los directorios de sujetos
    sub_dirs = sorted(bids_root.glob("sub-*"))
    if not sub_dirs:
        raise FileNotFoundError(f"No se encontraron directorios de sujetos (sub-*) en: {bids_root}")

    print(f"[INFO] Encontrados {len(sub_dirs)} sujetos para procesar")

    for sub_dir in sub_dirs:
        if not sub_dir.is_dir():
            continue
            
        print(f"\n[INFO] {'='*50}")
        print(f"[INFO] Procesando sujeto: {sub_dir.name}")
        print(f"[INFO] {'='*50}")

        eeg_dir = sub_dir / "eeg"
        if not eeg_dir.exists():
            print(f"[WARN] No existe directorio EEG para {sub_dir.name}, saltando...")
            continue

        out_dir = deriv_root / sub_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)

        set_files = sorted(eeg_dir.glob("*_eeg.set"))
        if not set_files:
            print(f"[WARN] No hay archivos .set en {eeg_dir}, saltando...")
            continue

        for f in set_files:
            try:
                site, cond = parse_site_cond_from_name(f.name)
                print(f"[INFO] Procesando {f.name} → site={site} cond={cond}")

                raw = read_raw_eeglab(f, preload=True, verbose=False)
                # --- [El resto del código de procesamiento permanece igual hasta el guardado] ---
                raw.rename_channels(_normalize_name)
                raw.pick_types(eeg=True, eog=False, ecg=False, emg=False, stim=False, misc=False)
                mont = make_standard_montage('brainproducts-RNP-BA-128')
                raw.set_montage(mont, on_missing='ignore')
                
                # --- DIAGNÓSTICO (código existente) ---
                print("\n[DIAG] ===== Montaje y posiciones asignadas =====")
                
                
                raw = _downsample_if_needed(raw, target_sfreq=1000.0)

                raw_clean, icas = preprocess_freedberg_mne(
                    raw,
                    montage='standard_1020',
                    interp_pulse=(-0.001, 0.015),
                    notch=60.0, notch_width=4.0,
                    l_freq=1.0, h_freq=100.0,
                    resample_sfreq=1000,
                    kurt_thresh=None,
                    do_csd=False
                )

                epochs = epochs_from_annotations(raw_clean, tmin=-0.1, tmax=0.5, baseline=(-0.1, -0.01))
                evk = epochs.average()
                evk.comment = f"{site}_{cond}"
                #evk_csd = compute_current_source_density(evk.copy())
                evk_csd = evk
                # Guardar para este sujeto
                out_fif = out_dir / f"{sub_dir.name}_{site}_{cond}_ave.fif"
                mne.write_evokeds(out_fif, evk_csd, overwrite=True)
                print(f"[OK] Guardado Evoked: {out_fif}")

            except Exception as e:
                print(f"[ERROR] Falló procesamiento de {f}: {e}")
                finite_cnt = 0
                zero_cnt = 0
                nan_cnt = 0
                rows = []
                for ch in raw.info['chs']:
                    name = ch['ch_name']
                    loc = np.array(ch['loc'][:3], dtype=float)
                    is_finite = np.all(np.isfinite(loc))
                    is_zero = np.allclose(loc, 0.0)
                    if not is_finite:
                        nan_cnt += 1
                    elif is_zero:
                        zero_cnt += 1
                    else:
                        finite_cnt += 1
                    rows.append(f"{name:>5s}  loc={loc}  finite={is_finite}  zero={is_zero}")

                # Imprime todas las filas (puedes comentar si es demasiado verboso)
                print("[DIAG] Canales y sus loc:")
                for r in rows:
                    print("[DIAG]  " + r)

                print(f"[DIAG] Resumen loc → finite_pos={finite_cnt}  zero_pos={zero_cnt}  nan_pos={nan_cnt}  total={len(raw.ch_names)}")

                # Mostrar 'bads' actuales en este punto
                print(f"[DIAG] BADS actuales ({len(raw.info.get('bads', []))}): {raw.info.get('bads', [])}")
                print("[DIAG] ==========================================\n")
                continue

if __name__ == "__main__":
    main()
