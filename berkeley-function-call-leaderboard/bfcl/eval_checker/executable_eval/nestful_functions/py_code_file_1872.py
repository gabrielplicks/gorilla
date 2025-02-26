from typing import List



def read_strings(file_path: str) -> List[str]:

    """Reads a list of strings from a file.



    Args:

        file_path: The path to the file.



    Returns:

        A list of strings read from the file.

    """

    with open(file_path, 'r') as file:

        contents = file.read()

        lines = contents.splitlines()

        stripped_lines = [line.strip() for line in lines]

        return stripped_lines

