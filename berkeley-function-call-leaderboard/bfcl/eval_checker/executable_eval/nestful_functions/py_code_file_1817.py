from typing import Union



def read_file_to_utf8_str(file_path: str) -> Union[str, None]:

    """Reads and returns the content of a plain text file as an UTF-8 encoded string.

    Handles exceptional cases where the file does not exist and prints an error message.



    Args:

        file_path: The path to the file.



    Returns:

        The file content as an UTF-8 encoded string, or None if the file does not exist.

    """

    try:

        with open(file_path, 'r') as file:

            content = file.read()

        return content.encode('utf-8')

    except FileNotFoundError:

        print(f"File not found at '{file_path}'")

        return None

