from typing import List



def read_file_to_list(path: str) -> List[str]:

    """Reads a file and returns a list of strings, where each string is a line in the file.



    Args:

        path: The path to the file to read.



    Returns:

        A list of strings, where each string is a line in the file.

    """

    with open(path, 'rt') as f:

        lines = f.readlines()

    return lines

