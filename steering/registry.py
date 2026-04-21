"""Property name -> head index mapping and direction logic."""

PROPERTY_NAMES = [
    "swi", "tango", "net_charge", "pI", "iupred3",
    "iupred3_fraction_disordered", "shannon_entropy",
    "hydrophobic_patch_total_area", "hydrophobic_patch_n_large",
    "sap", "scm_positive", "scm_negative", "rg",
]

PROPERTY_TO_INDEX = {name: i for i, name in enumerate(PROPERTY_NAMES)}

# direction string -> sign multiplier for the objective scalar.
# maximize: we want to INCREASE the prediction, so the objective to maximise
#   is +prediction. Gradient of +prediction w.r.t. z points uphill.
# minimize: we want to DECREASE, so objective is -prediction.
DIRECTION_SIGN = {
    "maximize": 1.0,
    "minimize": -1.0,
    "target": 0.0,        # handled specially in guide.py (squared distance)
    "target_range": 0.0,  # handled specially in guide.py (squared hinge)
}


def validate_objective(obj: dict) -> None:
    """Raise ValueError if an objective dict is malformed."""
    prop = obj.get("property")
    if prop not in PROPERTY_TO_INDEX:
        raise ValueError(
            f"Unknown property '{prop}'. Valid: {list(PROPERTY_TO_INDEX.keys())}"
        )
    direction = obj.get("direction", "maximize")
    if direction not in DIRECTION_SIGN:
        raise ValueError(
            f"Unknown direction '{direction}'. Valid: {list(DIRECTION_SIGN.keys())}"
        )
    if direction == "target" and obj.get("target_value") is None:
        raise ValueError("direction='target' requires 'target_value' to be set")
    if direction == "target_range":
        tmin, tmax = obj.get("target_min"), obj.get("target_max")
        if tmin is None and tmax is None:
            raise ValueError(
                "direction='target_range' requires at least one of 'target_min' / 'target_max'"
            )
        if tmin is not None and tmax is not None and tmin > tmax:
            raise ValueError(
                f"direction='target_range' requires target_min <= target_max, got {tmin} > {tmax}"
            )
