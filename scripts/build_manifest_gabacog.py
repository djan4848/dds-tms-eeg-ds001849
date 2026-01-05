#!/usr/bin/env python3
from pathlib import Path
import re
import pandas as pd
from configs.gabacog_sici_icf import DIR_SICI, DIR_ICF, MAP_SICI, MAP_ICF

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
        rows.append(dict(subject_id=sid, protocol=protocol, group=group, fif_path=str(p)))
    return rows

def main():
    rows = []
    rows += scan(DIR_SICI, "SICI", MAP_SICI)
    rows += scan(DIR_ICF,  "ICF",  MAP_ICF)

    df = pd.DataFrame(rows).sort_values(["protocol","group","subject_id"])
    out = Path("derivatives") / "dds_gabacog"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "manifest_gabacog_sici_icf.csv", index=False)

    # Reporta asimetrías para within
    sici = set(df.loc[df.protocol=="SICI","subject_id"])
    icf  = set(df.loc[df.protocol=="ICF","subject_id"])
    print("Only SICI:", sorted(sici-icf))
    print("Only ICF :", sorted(icf-sici))
    print("Within N :", len(sici & icf))
    print("Saved:", out / "manifest_gabacog_sici_icf.csv")

if __name__ == "__main__":
    main()

