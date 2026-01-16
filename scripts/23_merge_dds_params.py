"""
21_merge_dds_params.py

Une parámetros DDS early y late en un único CSV canónico:
  outputs/exports/dds/dds_params_all.csv

Incluye columna:
  window ∈ {early, late}
"""

from pathlib import Path
import pandas as pd

EARLY = Path("outputs/exports/dds/early/dds_params_15_80ms.csv")
LATE  = Path("outputs/exports/dds/late/dds_params_80_200ms.csv")
OUT   = Path("outputs/exports/dds/dds_params_all.csv")

missing = [p for p in [EARLY, LATE] if not p.exists()]
if missing:
    raise FileNotFoundError(f"Missing files: {missing}")

df_e = pd.read_csv(EARLY)
df_e["window"] = "early"

df_l = pd.read_csv(LATE)
df_l["window"] = "late"

df = pd.concat([df_e, df_l], ignore_index=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)

print(f"[OK] Wrote: {OUT}")
print(f"[OK] rows={len(df)} cols={len(df.columns)}")
print("[OK] windows:", df["window"].value_counts().to_dict())
print("[OK] columns:", list(df.columns))

