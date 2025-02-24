import numpy as np



def calculate_rss(observed_values: np.ndarray, predicted_values: np.ndarray) -> int:

    """Calculates the residual sum of squares (RSS) for a given set of real values.



    Args:

        observed_values: A NumPy array representing the observed values.

        predicted_values: A NumPy array representing the predicted values.



    Returns:

        The sum of squared differences between the observed values and the predicted values, rounded to the nearest integer.

    """

    differences = observed_values - predicted_values

    squared_differences = np.square(differences)

    sum_of_squared_differences = np.sum(squared_differences)

    return np.rint(sum_of_squared_differences)

