from typing import TextIO



def count_non_empty_lines(file_path: str) -> int:

    """Counts the number of non-empty lines in a file.



    Args:

        file_path: The path to the file.



    Returns:

        The number of non-empty lines in the file.

    """

    num_non_empty_lines = 0

    with open(file_path, 'r') as file:

        for line in file:

            if line.strip():

                num_non_empty_lines += 1

    return num_non_empty_lines

