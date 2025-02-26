from typing import Union



def extract_body(path: str) -> Union[str, None]:

    """Extracts the "body" or "text" of a text file at a given path.



    Args:

        path: A string representing the file path to the text file.



    Returns:

        The content of the file as a string, or None if the file is empty.

    """

    with open(path, 'r') as fin:

        text = fin.read()

    return text if text else None

