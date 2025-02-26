from typing import Tuple



def get_rows_and_columns(matrix_string: str) -> Tuple[int, int]:

    """Gets the number of rows and columns of a matrix from a string.



    The string is in matrix format, where each row is separated by a semicolon and each column by a space.

    The matrix is assumed to be rectangular and all cells are distinct.



    Args:

        matrix_string: The string representing the matrix.



    Returns:

        A tuple containing the number of rows and columns.

    """

    rows = matrix_string.split(";")

    num_rows = len(rows)



    num_columns = 0

    for row in rows:

        columns = row.split()

        num_columns = max(num_columns, len(columns))



    return num_rows, num_columns

