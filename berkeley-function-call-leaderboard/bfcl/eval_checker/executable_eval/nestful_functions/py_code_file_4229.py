from typing import TextIO



def count_comments(file_path: str) -> int:

    """Counts the number of single-line comments in a Python file.



    Args:

        file_path: The path to the Python file.



    Returns:

        The count of single-line comments in the file.

    """

    with open(file_path) as f:

        count = 0

        for line in f:

            if line.startswith('#'):

                count += 1

    return count

