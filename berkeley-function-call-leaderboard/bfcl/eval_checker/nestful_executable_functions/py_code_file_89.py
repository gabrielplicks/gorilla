from typing import List



def read_file_into_list(filename: str) -> List[str]:

    """Reads a file into a list of strings, ignoring empty lines and lines that start with a hash sign (`#`).



    Args:

        filename: The name of the file to read.



    Returns:

        A list of strings, where each string is a line from the file that is not empty and does not start with a hash sign.

    """

    with open(filename, 'r') as file:

        lines = [line.strip() for line in file]

        lines = [line for line in lines if line and not line.startswith('#')]

        return lines

