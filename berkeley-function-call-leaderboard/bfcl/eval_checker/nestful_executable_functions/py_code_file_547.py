from typing import List



def count_rows_with_target(data: List[List[int]], target: int) -> int:

    """Counts the number of rows in a matrix that contain a target integer.



    Args:

        data: A list of lists of integers representing a matrix.

        target: The target integer to search for in the matrix.



    Raises:

        TypeError: If the input data is not a list of lists of integers.

    """

    if not isinstance(data, list) or any(not isinstance(row, list) or any(not isinstance(item, int) for item in row) for row in data):

        raise TypeError("Invalid input: data must be a list of lists of integers")



    count = 0

    for row in data:

        if target in row:

            count += 1



    return count

