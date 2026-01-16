# DDS + Information Theory Pipeline (ds001849)

This repository implements a fully reproducible pipeline to study
information-theoretic properties of TMS-evoked EEG responses using
the DDS framework.

## Pipeline order

### 0. Environment
- scripts/00_check_env.py
- scripts/10_paths_and_artifacts.py

### 1. Preprocessing
- scripts/legacy/preprocess_ds001849_from_sets.py
Outputs: outputs/exports/evokeds/

### 2. DDS fitting
- scripts/21_run_dds_early.py
- scripts/22_run_dds_late.py
- scripts/23_merge_dds_params.py
Outputs: outputs/exports/dds/dds_params_all.csv

### 3. Information-theoretic features
- scripts/30_compute_entropy.py
- scripts/31_compute_mi_roi.py
- scripts/32_merge_mi_into_dds.py
- scripts/50_compute_mi_channelwise.py
Outputs: outputs/features/

### 4. Statistics
- scripts/40_stats_entropy_vs_dds.py
- scripts/41_stats_mi_vs_dds.py
- scripts/42_cond_sham_vs_active_m1.py
- scripts/43_cond_sham_vs_active_by_roi.py
- scripts/51_channelwise_cond_effects.py
Outputs: outputs/stats/

### 5. Sensor-level cluster tests
- scripts/60_cluster_perm_m1.py
- scripts/61_tfce_perm_m1.py
Outputs: outputs/stats/cluster_mi_tfce/, outputs/figures/tfce/

## Notes
- All analyses are within-subject.
- No notebooks are used.
- Legacy scripts are kept for traceability.
