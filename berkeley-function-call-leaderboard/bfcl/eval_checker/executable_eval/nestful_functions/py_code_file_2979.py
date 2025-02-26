import numpy as np

from typing import List

from typing import List



def center_and_covariance(ts_data: List[List[float]]) -> np.ndarray:

    """Calculates the covariance matrix of a list of time series data after centering each series.



    Args:

        ts_data: A list of time series data, where each series is a list of float values.



    Returns:

        The covariance matrix of the centered time series data.

    """

    centered_data = []

    for series in ts_data:

        mean = sum(series) / len(series)

        centered_data.append([x - mean for x in series])

    return np.cov(centered_data)

