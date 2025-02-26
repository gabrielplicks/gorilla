import os.path



def get_directory_name(file_name: str) -> str:

    """

    Returns the directory name of a file, excluding the file name itself.

    Args:

        file_name: The name of the file.

    Raises:

        AssertionError: If the input argument `file_name` is not a valid file name.

    """

    if not file_name or any(char in file_name for char in ['*', '?', '"', '<', '>']):

        raise AssertionError("Invalid file name")

    directory_name = os.path.dirname(file_name)



    return directory_name

