"""
00_check_env.py

Sanity check para repos basados en scripts (sin src/ y sin paquete instalable).
Este script NO procesa datos.
Verifica:
1) versión de Python
2) que existen carpetas esperadas (scripts/, notebooks/)
3) que requirements están presentes
4) crea (si no existe) la estructura mínima src/ para el módulo info-theory
"""

from pathlib import Path
import sys

print("=" * 60)
print("ENVIRONMENT SANITY CHECK (script-only repo)")
print("=" * 60)

print(f"Python: {sys.version}")

ROOT = Path(__file__).resolve().parents[1]
print(f"Repo root: {ROOT}")

# Check expected dirs/files
expected = [
    ("scripts", (ROOT / "scripts").is_dir()),
    ("notebooks", (ROOT / "notebooks").is_dir()),
    ("README.md", (ROOT / "README.md").is_file()),
    ("requirements.txt", (ROOT / "requirements.txt").is_file()),
    ("requirements_ds001849.txt", (ROOT / "requirements_ds001849.txt").is_file()),
]
for name, ok in expected:
    print(f"{name:>26}: {'OK' if ok else 'MISSING'}")

# Prepare minimal src layout for our new code (info-theory layer)
SRC = ROOT / "src" / "dds_it"
(SRC / "info").mkdir(parents=True, exist_ok=True)
(SRC / "utils").mkdir(parents=True, exist_ok=True)

# Ensure packages are packages
for p in [ROOT / "src", ROOT / "src" / "dds_it", SRC / "info", SRC / "utils"]:
    init = p / "__init__.py"
    if not init.exists():
        init.write_text("# Auto-created to enable imports\n")

print(f"Prepared minimal package skeleton at: {ROOT / 'src' / 'dds_it'}")

print("=" * 60)
print("Sanity check completed")
print("=" * 60)

