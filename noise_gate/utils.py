import math


def db_to_linear(db: float) -> float:
    """Convert decibel value to linear scale."""
    return 10 ** (db / 20)


def linear_to_db(x: float) -> float:
    """Convert linear value to decibels.

    The input value is clamped to a minimum of 1e-10 to avoid log(0).
    """
    x = max(x, 1e-10)
    return 20 * math.log10(x)
