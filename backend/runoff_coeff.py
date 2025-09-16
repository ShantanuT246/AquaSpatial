# runoff_strict.py
_RUNOFF_MAP = {
    "metal": 0.95,
    "concrete": 0.95,
    "asphalt": 0.90,
    "tile": 0.80,
}

def get_runoff_coefficient_strict(roof_type: str) -> float:
    key = roof_type.strip().lower()
    if key not in _RUNOFF_MAP:
        raise ValueError("Allowed: tile, metal, asphalt, concrete")
    return float(_RUNOFF_MAP[key])
