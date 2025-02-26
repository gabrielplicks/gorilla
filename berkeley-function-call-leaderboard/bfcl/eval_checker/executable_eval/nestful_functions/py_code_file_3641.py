import numpy as np



def spherical_harmonic_values(theta: float, phi: float) -> float:

    """Calculates the spherical harmonic values for the given theta and phi.



    Args:

        theta: The theta value.

        phi: The phi value.



    Returns:

        The spherical harmonic value.

    """

    l = 2

    m = 0

    x = np.cos(theta)

    P_l_m = 0.5 * (3 * x**2 - 1)

    Y_l_m = np.sqrt((2*l+1)/(4*np.pi)) * P_l_m



    return Y_l_m

