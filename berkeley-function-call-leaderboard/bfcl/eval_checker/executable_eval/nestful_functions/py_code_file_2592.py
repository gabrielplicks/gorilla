from typing import List, Union



def compare_vectors(x: List[Union[int, float]], y: List[Union[int, float]]) -> bool:

    """Compares two vectors and returns True if they are the same, False otherwise.



    Args:

        x: The first vector.

        y: The second vector.



    Returns:

        True if the two vectors are the same, False otherwise.

    """

    for x_elem, y_elem in zip(x, y):

        if x_elem != y_elem:

            return False

    return True

