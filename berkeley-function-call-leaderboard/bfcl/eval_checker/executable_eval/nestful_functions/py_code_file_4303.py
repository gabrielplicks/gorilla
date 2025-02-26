import math

import numpy as np



def information_gain(p: np.ndarray) -> float:

    """Calculates the expected information gain of a categorical attribute with a given probability distribution.



    Args:

        p: A 1D NumPy array or a list of probabilities representing the probability distribution of the categorical attribute.



    Returns:

        The expected information gain as a float. If NaNs or infinite values are found in the input `p`, the function returns a NaN.

    """

    if np.isnan(p).any() or np.isinf(p).any():

        return np.nan

    entropy = -np.sum(p * np.log2(p))

    expected_info_gain = np.sum(p * entropy)



    return expected_info_gain

