from typing import List



def find_positive_columns(mat: List[List[int]]) -> List[int]:

    """Finds the indices of columns in a 2D list that contain at least one positive integer.



    Args:

        mat: A 2D list of integers.



    Returns:

        A list of column indices.

    """

    col_idx = [i for i, col in enumerate(zip(*mat)) if any(c > 0 for c in col)]

    return col_idx

