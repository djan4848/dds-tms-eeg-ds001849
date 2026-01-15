"""
10_paths_and_artifacts.py

Paso 1: inspección del entorno y detección de artefactos existentes.
- Verifica DS001849_ROOT
- Verifica outputs/
- Busca evokeds exportados (si existen)
- Busca DDS params exportados (si existen)
"""

from pathlib import Path
import sys

# Permitir imports desde src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dds_it.utils.paths import paths  # noqa: E402

print("=" * 70)
print("PATHS & ARTIFACTS CHECK")
print("=" * 70)
print("Repo root     :", paths.repo_root)
print("DS001849 root :", paths.ds_root)
print("Outputs root  :", paths.outputs_root)

# Ensure outputs dirs exist
for p in [paths.outputs_root, paths.exports_root, paths.features_root, paths.stats_root, paths.figures_root]:
    p.mkdir(parents=True, exist_ok=True)

print("\nOutputs dirs OK:")
print(" -", paths.exports_root)
print(" -", paths.features_root)
print(" -", paths.stats_root)
print(" -", paths.figures_root)

# Heurística: dónde podrían estar evokeds / params DDS
# (sin asumir nombres exactos todavía)
candidate_patterns = [
    ("Evokeds (fif)", "**/*-ave.fif"),
    ("Evokeds (fif)", "**/*ave.fif"),
    ("Evokeds (npz)", "**/*.npz"),
    ("Evokeds (pkl)", "**/*.pkl"),
    ("DDS params (csv)", "**/*dds*.csv"),
    ("DDS params (parquet)", "**/*dds*.parquet"),
    ("DDS params (npz)", "**/*dds*.npz"),
]

search_roots = [
    paths.outputs_root,
    paths.repo_root,
]

print("\nSearching for candidate artifacts (quick scan)...")
found_any = False
for label, pattern in candidate_patterns:
    matches = []
    for r in search_roots:
        matches.extend(list(r.glob(pattern)))
    matches = sorted(set(matches))[:10]
    if matches:
        found_any = True
        print(f"\n{label} | pattern={pattern} | showing up to 10:")
        for m in matches:
            print(" -", m)

if not found_any:
    print("\nNo artifacts found yet (OK if this is a fresh checkout).")

print("\nNEXT:")
print("1) If ds001849 is not inside repo/data/ds001849, set:")
print("   export DS001849_ROOT=/path/to/ds001849")
print("2) Then we will run preprocessing and DDS fitting (if needed) to generate artifacts.")
print("=" * 70)

