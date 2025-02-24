from typing import List



def is_valid_coordinates(matrix: List[List[str]]) -> bool:

    """Checks if the values in a 2D list of strings are valid coordinates.



    Args:

        matrix: A 2D list of strings representing a matrix.



    Returns:

        A boolean value indicating whether the values in the matrix are valid coordinates.

    """

    for row in matrix:

        for element in row:

            try:

                number = int(element)

                if not (-100 <= number <= 100):

                    return False

            except ValueError:

                return False

    return True

