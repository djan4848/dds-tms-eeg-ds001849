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
    "DLPFC_R": ["AF4","F4","F6","F2","FC4","FC6"],
    "DLPFC_L": ["AF3","F3","F5","F1","FC3","FC5"],
    "FRONTAL_MID": ["Fpz","Fz","FC1","FC2"],
    "SENSORIMOTOR_L": ["C3","C1","C5","CP3","CP1","CP5"],
    "SENSORIMOTOR_R": ["C4","C2","C6","CP4","CP2","CP6"],
    "PARIETAL_MID": ["Pz","CPz","P1","P2","POz"],
    "PARIETAL_L": ["P3","P5","P7","PO3","PO7"],
    "PARIETAL_R": ["P4","P6","P8","PO4","PO8"],
    "TEMPORAL": ["T7","T8","FT7","FT8","TP7","TP8"],
    "OCCIPITAL": ["O1","O2","Oz","PO7","PO8"],
    




    # 1) Motor / peri-motor (ya lo tenías; sin FCz)
    "MOTOR_FC_C": ["FC3","FC1","C3","C1","Cz","CP3","CP1","CPz"],

    # 2) DLPFC izquierda (aprox. F3/FC3/AF3; incluyo F1/Fz como “dorsal” si están)
    "DLPFC_L": ["AF3","F3","F5","F1","FC3","FC5"],

    # 3) DLPFC derecha (simétrico)
    "DLPFC_R": ["AF4","F4","F6","F2","FC4","FC6"],

    # 4) Parietal medial (red más “posterior”; útil si hay efectos de red)
    "PARIETAL_MID": ["Pz","CPz","P1","P2","CP1","CP2"],
}

