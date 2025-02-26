import numpy as np



def compute_total_size_in_bytes(matrix: np.ndarray, dtype: str) -> int:

    """Computes the total size of the elements in a 2D array (matrix) in bytes.



    Args:

        matrix: The 2D array.

        dtype: The data type of the elements.



    Returns:

        The total size in bytes of all the elements in the matrix.

    """

    dtype_info = np.dtype(dtype)

    num_bytes_per_element = dtype_info.itemsize

    num_elements = np.size(matrix)

    total_size_in_bytes = num_bytes_per_element * num_elements

    return total_size_in_bytes

