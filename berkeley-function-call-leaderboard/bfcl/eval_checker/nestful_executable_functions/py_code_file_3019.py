from typing import TextIO



def count_lines_of_code(file_path: str) -> int:

    """Counts the total number of lines in a given file, skipping empty lines and lines that begin with a comment symbol.



    Args:

        file_path: The path to the file.



    Returns:

        The total number of lines in the file.

    """

    num_lines = 0

    with open(file_path, 'r') as file:

        for line in file:

            line = line.strip()  # Remove leading and trailing whitespace

            if line and not line.startswith('#'):  # Skip empty lines and comment lines

                num_lines += 1

    return num_lines

