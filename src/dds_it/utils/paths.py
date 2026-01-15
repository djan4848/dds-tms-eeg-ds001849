"""
paths.py

Centraliza rutas para que no dependan del working directory.
NO asume nada del sistema salvo variables de entorno opcionales.

Uso:
  from dds_it.utils.paths import paths
  print(paths.ds_root)
"""

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(frozen=True)
class Paths:
    repo_root: Path
    ds_root: Path
    outputs_root: Path
    exports_root: Path
    features_root: Path
    stats_root: Path
    figures_root: Path


def _env_path(var: str) -> Path | None:
    v = os.environ.get(var, "").strip()
    return Path(v).expanduser().resolve() if v else None


def make_paths() -> Paths:
    repo_root = Path(__file__).resolve().parents[3]

    # Datasets: por defecto buscamos una carpeta "data/ds001849" dentro del repo.
    # Puedes sobreescribir con: export DS001849_ROOT=/ruta/al/dataset
    ds_root = _env_path("DS001849_ROOT")
    if ds_root is None:
        ds_root = (repo_root / "data" / "ds001849").resolve()

    outputs_root = (repo_root / "outputs").resolve()
    exports_root = outputs_root / "exports"
    features_root = outputs_root / "features"
    stats_root = outputs_root / "stats"
    figures_root = outputs_root / "figures"

    return Paths(
        repo_root=repo_root,
        ds_root=ds_root,
        outputs_root=outputs_root,
        exports_root=exports_root,
        features_root=features_root,
        stats_root=stats_root,
        figures_root=figures_root,
    )


paths = make_paths()

