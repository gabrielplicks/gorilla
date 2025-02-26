import numpy as np



def detect_outliers_iqr(data: np.ndarray) -> np.ndarray:

    """Detects and removes outliers from a NumPy array using the interquartile range (IQR) method.



    Args:

        data: The input NumPy array.



    Returns:

        A new NumPy array containing the original values with outliers removed.

    """

    q1 = np.percentile(data, 25)

    q3 = np.percentile(data, 75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr

    upper_bound = q3 + 1.5 * iqr

    return data[np.logical_and(data >= lower_bound, data <= upper_bound)]

