from typing import Union



class InvalidFilePath(Exception):

    pass



def get_file_basename(file_path: Union[str, None]) -> str:

    """

    Returns the base name of a file given its path.

    Args:

        file_path: The path of the file.

    Raises:

        InvalidFilePath: If the file path is invalid.

    """

    if not file_path:

        raise InvalidFilePath("The file path cannot be empty.")

    file_path = file_path.strip()

    if not file_path:

        raise InvalidFilePath("The file path cannot consist only of whitespace characters.")

    if not isinstance(file_path, str):

        raise InvalidFilePath("The file path must be a string.")

    basename = file_path.split('/')[-1]

    return basename

