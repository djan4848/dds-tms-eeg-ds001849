# dds-tms-eeg-ds001849
Code and notebooks for DDS model validation and comparison with cosine similarity on TMS–EEG (OpenNeuro ds001849)
# DDS TMS-EEG (ds001849) — Pipeline mínimo

Este repositorio contiene scripts para el ajuste del modelo **Dual Damped Sine (DDS)** sobre TMS-EEG (dataset **OpenNeuro ds001849**), con análisis por ventanas **15–80 ms** y **80–200 ms**.

## Requisitos
python 3.10
numpy, pandas, scipy, statsmodels, mne, matplotlib
Opcional: `mne-bids`, `joblib`

Instalación rápida:
```bash
pip install -r requirements.txt
# (opcional) pip install -r requirements_ds001849.txt

Script principal
scripts/run_dds_from_evokeds_new.py

Descripción

Carga evokeds por sujeto/sitio/condición.

Ajusta DDS en dos ventanas: 15–80 ms (temprana) y 80–200 ms (tardía).

Exporta parámetros (A1, γ1, f1, A2, γ2, f2) y métricas de ajuste (R², RMSE).

Genera tablas CSV y figuras de resumen (opcional).

Uso típico

python scripts/run_dds_from_evokeds_new.py \
  --deriv_root /ruta/a/derivatives/mne_freedberg \
  --sites m1 dlpfc ppc \
  --windows 15-80 80-200 \
  --out_dir ./stats_results \
  --make_figs


Argumentos clave

--deriv_root: raíz con carpetas sub-XX y ficheros *_ave.fif.

--sites: sitios a procesar (ej. m1 dlpfc ppc).

--windows: ventanas a evaluar (15-80, 80-200).

--out_dir: carpeta de salida para CSV/figuras.

--make_figs: si se incluye, genera violin plots y métricas.

Salida

stats_results/dds_params_15_80ms.csv

stats_results/dds_params_80_200ms.csv

(opcional) stats_results/figs/*.png

Notas

Las ventanas siguen el criterio de Freedberg (early/late).

.gitignore excluye datos/derivados por defecto (derivatives/, stats_results/, etc.).
