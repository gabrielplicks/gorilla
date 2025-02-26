from typing import Union



def split_extension(path: str) -> Union[str, None]:

    """Extracts the file extension from a path.



    Args:

        path: The file path.



    Returns:

        The file extension, or None if the path does not have a dot (`.`) character.

    """

    return path.split('.')[-1]

