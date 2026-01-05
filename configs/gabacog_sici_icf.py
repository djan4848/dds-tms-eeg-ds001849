from pathlib import Path

# Ajusta a tu disco
ROOT = Path("/media/neuraldyn/Extreme SSD/GABACOG")
DIR_SICI = ROOT / "procesado_SICI"
DIR_ICF  = ROOT / "procesado_ICF"

# Sujetos por grupo (tal como lo diste)
SICI_CONTROLES = {'005','006','008','009','010','012','013','014','016','017','018','019','020','021','022','023','024','028','039'}
SICI_TOC       = {'027','029','030','031','032','033','036','038','040','041','042','043','044','045','046','047','048','049','050'}

ICF_CONTROLES  = {'005','006','008','009','010','012','013','014','016','017','018','019','020','021','022','023','024','028','039','035'}
ICF_TOC        = {'027','029','030','031','032','036','038','040','041','042','043','044','045','046','047','048','049','050'}

def build_map(controls, toc):
    m = {s: "CTL" for s in controls}
    m.update({s: "TOC" for s in toc})
    overlap = set(controls) & set(toc)
    if overlap:
        raise ValueError(f"Overlap CTL/TOC: {sorted(overlap)}")
    return m

MAP_SICI = build_map(SICI_CONTROLES, SICI_TOC)
MAP_ICF  = build_map(ICF_CONTROLES,  ICF_TOC)

# Ventanas de ajuste DDS (según tu paper SICI/ICF)
WINDOWS = {
    "SICI": [(0.010, 0.100, "10_100ms")],
    "ICF":  [(0.010, 0.200, "10_200ms")],
}

# Baseline: tus epochs traen baseline definido (-0.5, 0.0) pero no aplicado => aplicarlo explícito
BASELINE = (-0.5, 0.0)

# ROI secundario (edítalo a tu gusto)
ROI = {
    "MOTOR_FC_C": ["FC3","FC1","FCz","C3","C1","Cz","CP3","CP1","CPz"]
}

