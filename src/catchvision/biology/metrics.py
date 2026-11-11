from __future__ import annotations

import numpy as np


def estimate_weight_g(length_cm: float, lwr_a: float, lwr_b: float) -> float:
    """Estimate body weight (g) from the allometric equation W = a * L^b."""
    if length_cm <= 0:
        raise ValueError("length_cm must be positive.")
    return float(lwr_a * (length_cm**lwr_b))


def fulton_condition_factor(weight_g: float, length_cm: float) -> float:
    """Compute Fulton's condition factor K = 100 * W / L^3."""
    if length_cm <= 0:
        raise ValueError("length_cm must be positive.")
    return float(100.0 * weight_g / (length_cm**3))


def length_weight_curve(
    lwr_a: float,
    lwr_b: float,
    min_cm: float = 10.0,
    max_cm: float = 70.0,
    n_pts: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a length-weight curve for EDA plots."""
    lengths = np.linspace(min_cm, max_cm, n_pts)
    weights = lwr_a * (lengths**lwr_b)
    return lengths, weights
