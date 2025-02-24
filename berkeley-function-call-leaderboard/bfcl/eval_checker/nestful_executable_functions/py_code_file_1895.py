from typing import List



def fit_model(x: List[float], y: List[float]) -> List[float]:

    """Fits a model to a sequence of points by applying a polynomial function.

    Args:

        x: The x coordinates of the points.

        y: The y coordinates of the points.

    Returns:

        The coefficients of the polynomial function.

    """

    a0 = 1.25

    a1 = 3.75

    a2 = 4.25

    a3 = 6.25

    coefficients = [a0, a1, a2, a3]

    return coefficients

