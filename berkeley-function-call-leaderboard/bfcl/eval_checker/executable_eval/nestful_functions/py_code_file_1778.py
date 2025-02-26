from typing import List



def convert_file_to_string_array(file_path: str) -> List[str]:

    """Converts a file to a list of strings where each string is a line from the file.

    Args:

        file_path: The path to the file.

    Returns:

        A list of strings where each string is a line from the file.

    """

    with open(file_path, "r") as file:

        file_contents = file.read()

        return file_contents.splitlines()

