import numpy as np

from typing import Dict



def fit_polynomial(x: np.ndarray, y: np.ndarray, order: int) -> Dict[int, float]:

    """Fits a polynomial curve of specified order to a set of points (x, y) using linear regression.



    Args:

        x: The x-coordinates of the points.

        y: The y-coordinates of the points.

        order: The order of the polynomial to fit.



    Returns:

        A dictionary of coefficients where keys are powers of x and values are the corresponding coefficients.

    """

    X = np.vander(x, order+1)

    a = np.linalg.lstsq(X, y, rcond=None)[0]

    return dict(zip(range(order+1), a))

