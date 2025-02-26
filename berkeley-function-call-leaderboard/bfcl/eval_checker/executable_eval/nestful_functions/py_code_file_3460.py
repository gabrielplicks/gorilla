import math



def standard_gaussian_pdf(x: float) -> float:

    """Calculates the probability density function (PDF) value for a standard Gaussian distribution.



    Args:

        x: The value for which the PDF is calculated.



    Returns:

        The PDF value for the given x.

    """

    return 1 / math.sqrt(2 * math.pi) * math.exp(-(x ** 2) / 2)

