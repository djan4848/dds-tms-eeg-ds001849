#!/usr/bin/env python3
from pathlib import Path
import re
import pandas as pd
from configs.gabacog_sici_icf import DIR_SICI, DIR_ICF, MAP_SICI, MAP_ICF, WINDOWS

PAT = re.compile(r"sub_(\d{3})__([A-Z]+)-epo\.fif$")

def scan(dirpath: Path, protocol: str, mapping: dict):
    rows = []
    for p in sorted(dirpath.glob("sub_*__*-epo.fif")):
        m = PAT.search(p.name)
        if not m:
            continue
        sid, proto = m.group(1), m.group(2)
        if proto != protocol:
            continue

        group = mapping.get(sid, "NA")

        # Si hay más de una ventana por protocolo, puedes replicar filas aquí;
        # por ahora usas 1 ventana por protocolo (como en tu config).
        tmin, tmax, wlab = WINDOWS[protocol][0]

        rows.append(dict(
            subject_id=sid,
            protocol=protocol,
            group=group,
            window=wlab,
            tmin=tmin,
            tmax=tmax,
            fif_path=str(p),
        ))
    return rows

def main():
    rows = []
    rows += scan(DIR_SICI, "SICI", MAP_SICI)
    rows += scan(DIR_ICF,  "ICF",  MAP_ICF)

    df = pd.DataFrame(rows).sort_values(["protocol","group","subject_id","channel"] if "channel" in rows else ["protocol","group","subject_id"])

    out = Path("derivatives") / "dds_gabacog"
    out.mkdir(parents=True, exist_ok=True)
    out_csv = out / "manifest_gabacog_sici_icf.csv"
    df.to_csv(out_csv, index=False)

    # Reporte útil (sin emparejar ni filtrar)
    print("\nManifest summary (no pairing enforced):")
    print(df.groupby(["protocol","group"])["subject_id"].nunique().rename("n_subjects"))

    na = df[df["group"]=="NA"]["subject_id"].unique().tolist()
    if na:
        print("\n[WARN] Subjects with group==NA (not in mapping):", sorted(na))

    print("\nSaved:", out_csv)

if __name__ == "__main__":
    main()

