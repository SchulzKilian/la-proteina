"""Guidance weight schedules w(t)."""
from __future__ import annotations

import math


def linear_ramp(t: float, t_start: float, t_end: float, w_max: float) -> float:
    if t < t_start:
        return 0.0
    if t >= t_end:
        return w_max
    return w_max * (t - t_start) / (t_end - t_start)


def cosine_ramp(t: float, t_start: float, t_end: float, w_max: float) -> float:
    if t < t_start:
        return 0.0
    if t >= t_end:
        return w_max
    progress = (t - t_start) / (t_end - t_start)
    return w_max * 0.5 * (1.0 - math.cos(math.pi * progress))


def constant(t: float, t_start: float, t_end: float, w_max: float) -> float:
    if t < t_start:
        return 0.0
    return w_max


_SCHEDULE_REGISTRY = {
    "linear_ramp": linear_ramp,
    "cosine_ramp": cosine_ramp,
    "constant": constant,
}


def get_schedule_fn(schedule_cfg: dict):
    """Return a callable t -> w(t) from a schedule config block."""
    stype = schedule_cfg["type"]
    if stype not in _SCHEDULE_REGISTRY:
        raise ValueError(f"Unknown schedule type '{stype}'. Valid: {list(_SCHEDULE_REGISTRY.keys())}")
    fn = _SCHEDULE_REGISTRY[stype]
    t_start = schedule_cfg.get("t_start", 0.3)
    t_end = schedule_cfg.get("t_end", 1.0)
    w_max = schedule_cfg.get("w_max", 2.0)
    return lambda t: fn(t, t_start, t_end, w_max)
