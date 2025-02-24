from typing import List



def detect_outliers(data: List[float]) -> List[int]:

    """Detects outliers in a dataset.



    An outlier is a data point that is significantly different from the rest of the data.

    This function uses the mean and standard deviation to detect outliers.



    Args:

        data: A list of numbers.



    Returns:

        A list of indices of the outliers.

    """

    mean = sum(data) / len(data)

    std = (sum((x - mean)**2 for x in data) / len(data))**0.5

    outlier_indices = []

    for i, x in enumerate(data):

        if abs(x - mean) > 2 * std:

            outlier_indices.append(i)



    return outlier_indices

