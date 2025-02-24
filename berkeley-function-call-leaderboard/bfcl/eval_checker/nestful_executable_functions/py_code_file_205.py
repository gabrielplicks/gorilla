import numpy as np



def redshift(v: float) -> float:

    """Computes the redshift (z), given the recessional velocity (v).

    Args:

        v: The recessional velocity in km/s.

    """

    c = 299792.458  # Speed of light in km/s

    if v < 0:

        return np.nan

    z = v / c

    return z

