import math



def gaussian_density(x: float, mu: float, sigma: float) -> float:

    """Calculates the density of a given value `x` with respect to a Gaussian distribution with mean `mu` and standard deviation `sigma`.

    If `x` is not in the range `[-3 * sigma, 3 * sigma]`, the function returns `0`.

    Args:

        x: The value to calculate the density for.

        mu: The mean of the Gaussian distribution.

        sigma: The standard deviation of the Gaussian distribution.

    """

    if x < -3 * sigma or x > 3 * sigma:

        return 0

    else:

        return (1 / (math.sqrt(2 * math.pi * sigma**2))) * math.exp(-((x - mu)**2) / (2 * sigma**2))

