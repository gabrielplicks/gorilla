from typing import List, Tuple, Union



def search_sparse_matrix(A: Union[List[List[float]], List[Tuple[int, int, float]]], x: float) -> bool:

    """Searches for a value `x` in a matrix `A`.



    Args:

        A: A matrix represented as a list of lists or a sparse matrix represented as a list of tuples.

        x: The value to search for in the matrix.



    Returns:

        True if `x` is present in `A`, False otherwise.

    """

    if type(A[0]) == tuple:  # If A is a sparse matrix

        for row, col, val in A:

            if val == x:

                return True

    else:  # If A is a normal matrix

        for row in A:

            for val in row:

                if val == x:

                    return True

    return False

