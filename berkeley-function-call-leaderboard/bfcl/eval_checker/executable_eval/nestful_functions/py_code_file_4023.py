from typing import List

import pathlib



def has_valid_extension(file_path: str, extensions: List[str]) -> bool:

    """Checks if a file has a valid extension based on a list of valid extensions.



    Args:

        file_path: A string representing the path to a file.

        extensions: A list of valid extensions.



    Returns:

        True if the file has a valid extension, False otherwise.

    """

    file_extension = pathlib.Path(file_path).suffix



    if file_extension in extensions:

        return True

    elif not file_extension and "" in extensions:

        return True

    elif not extensions:

        return True

    else:

        return False

