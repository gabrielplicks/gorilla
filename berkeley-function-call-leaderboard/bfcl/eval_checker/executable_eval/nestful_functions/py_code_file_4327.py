import math

from typing import List



def calculate_rmse(y_true: List[float], y_pred: List[float]) -> float:

    """Calculates the root mean square error (RMSE) between two lists of numbers.



    Args:

        y_true: The true values.

        y_pred: The predicted values.



    Raises:

        ValueError: If the lists have different lengths.

    """

    if len(y_true) != len(y_pred):

        raise ValueError("The lists must have the same length.")

    squared_diffs = [(a - b) ** 2 for a, b in zip(y_true, y_pred)]

    squared_diffs_sum = sum(squared_diffs)

    rmse = math.sqrt(squared_diffs_sum / len(y_true))

    return rmse

