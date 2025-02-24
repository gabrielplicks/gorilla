import numpy as np



def compare_matrices_with_mask(m1: np.ndarray, m2: np.ndarray, m_mask: np.ndarray) -> np.ndarray:

    """

    Compares the elements of two matrices element-wise using a boolean mask.



    The comparison result is True if the corresponding entry in the mask is True and the corresponding elements in the two matrices are equal, and False otherwise.



    Args:

        m1: The first matrix to compare.

        m2: The second matrix to compare.

        m_mask: The boolean mask indicating which elements to compare.



    Returns:

        A NumPy array with the same shape as the input matrices containing the comparison results.

    """

    assert m1.shape == m2.shape == m_mask.shape, "Matrices and mask must have the same shape"

    compare_mask = np.logical_and(m_mask, m1 == m2)



    return compare_mask

