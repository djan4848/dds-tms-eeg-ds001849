from pathlib import Path
import pandas as pd

DDS = Path("outputs/exports/dds/dds_params_all.csv")
MI  = Path("outputs/features/info_mutual_information.csv")
OUT = Path("outputs/features/dds_plus_mi.csv")

df = pd.read_csv(DDS)
mi = pd.read_csv(MI)

# merge por sujeto/sitio/cond (MI es una medida por combinación)
merged = df.merge(mi[["subject","site","cond","mi_early_late_bits","bins_early","bins_late","resample_n"]],
                  on=["subject","site","cond"], how="left")

OUT.parent.mkdir(parents=True, exist_ok=True)
merged.to_csv(OUT, index=False)

print("[OK] wrote:", OUT)
print("MI availability:", merged["mi_early_late_bits"].notna().mean())
