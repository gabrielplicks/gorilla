from typing import Union



def get_first_line_of_text_file_with_error_handling(file_path: str) -> Union[str, None]:

    """

    Reads the first line of a text file.



    Args:

        file_path: The path to the file.



    Returns:

        The first line of the file, or a string indicating that the file could not be read.

    """

    try:

        with open(file_path) as file:

            return file.readlines()[0]

    except FileNotFoundError:

        return "The file could not be read."

