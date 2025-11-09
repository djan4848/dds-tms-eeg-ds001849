#!/usr/bin/bash
# Process all subjects for both windows
for subject in 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 ; do
  echo "Processing subject $subject..."
  
  # 0-100ms window
  python run_dds_from_evokeds.py \
    --deriv_root ../derivatives/mne_freedberg \
    --subject $subject \
    --out_dir ../derivatives/dds_ds001849 \
    --tmin 0.0 --tmax 0.1 \
    --save_figs
  
  # 100-300ms window  
  python run_dds_from_evokeds.py \
    --deriv_root ../derivatives/mne_freedberg \
    --subject $subject \
    --out_dir ../derivatives/dds_ds001849 \
    --tmin 0.1 --tmax 0.2 \
    --save_figs
done
