"""Thin wrapper that reads `config.yaml`'s `production_information` and
resolves the Grade Selection profile into the dict historically consumed by
`Detector`.

The profile package lives at repo-level `src/Profiles/profiles/` and is shared
by all modules (DefectDetection, BalerClassification, Inspector-AIServer).
We bootstrap `sys.path` so `from profiles import ...` resolves without
requiring a global install.
"""
from __future__ import annotations
from profiles import resolve_from_file  # noqa: E402

def load_config(config_path: str = "config.yaml") -> dict:
    return resolve_from_file(config_path, section="defect_detection")