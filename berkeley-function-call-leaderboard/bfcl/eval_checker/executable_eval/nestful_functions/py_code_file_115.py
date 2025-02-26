import numpy as np



def rmse(actual: np.ndarray, forecast: np.ndarray) -> float:

    """Calculates the root mean squared error (RMSE) between two array-like objects.



    Args:

        actual: The actual values as an array-like object.

        forecast: The forecast values as an array-like object.



    Returns:

        The RMSE between the actual and forecast values.



    Raises:

        ValueError: If the actual and forecast arrays have different lengths.

    """

    if len(actual) != len(forecast):

        raise ValueError("The actual and forecast arrays must have the same length.")

    actual = np.ravel(actual)

    forecast = np.ravel(forecast)

    diff = actual - forecast

    squared_diff = diff ** 2

    mean = np.mean(squared_diff)

    return np.sqrt(mean)

