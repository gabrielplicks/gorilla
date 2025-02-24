import numpy as np



def compute_spearman_correlation(array1: np.ndarray, array2: np.ndarray) -> float:

    """Computes the Spearman's rank correlation coefficient between two arrays.



    Args:

        array1: A 1-dimensional NumPy array.

        array2: A 1-dimensional NumPy array of equal length to array1.



    Returns:

        A single float value representing the correlation coefficient.

    """

    array1 = np.squeeze(array1)

    array2 = np.squeeze(array2)

    if len(array1) != len(array2):

        raise ValueError('Arrays must be of equal length.')

    rank1 = np.argsort(np.argsort(array1))

    rank2 = np.argsort(np.argsort(array2))

    correlation = np.corrcoef(rank1, rank2)[0, 1]



    return correlation

