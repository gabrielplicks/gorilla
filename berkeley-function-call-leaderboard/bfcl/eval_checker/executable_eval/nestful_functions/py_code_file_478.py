import numpy as np



def fraction_positive(array: np.ndarray) -> float:

    """Calculates the fraction of positive values among all values in the NumPy array.



    Args:

        array: A NumPy array of integer values.



    """

    num_positive = np.count_nonzero(array > 0)

    total_num = array.size

    return num_positive / total_num

