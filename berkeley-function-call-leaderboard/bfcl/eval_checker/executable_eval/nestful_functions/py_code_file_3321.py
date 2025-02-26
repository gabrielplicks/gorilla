from typing import List



def convert_zigzag(s: str, num_rows: int) -> str:

    """Converts a string to the ZigZag format.



    Args:

        s: The input string.

        num_rows: The number of rows in the ZigZag format.



    Returns:

        The string after converting it to the ZigZag format.

    """

    if num_rows == 1:

        return s



    result: List[str] = [''] * num_rows



    row: int = 0

    direction: int = 1



    for c in s:

        result[row] += c

        row += direction



        if row == 0 or row == num_rows - 1:

            direction *= -1



    return ''.join(result)

