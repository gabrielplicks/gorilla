from typing import Tuple



def split_file_name(file_name: str) -> Tuple[str, str]:

    """Splits a file name into its base name and extension.



    If the file name does not have an extension, returns the file name as the base name and an empty string as the extension.



    Args:

        file_name: The file name to split.



    Returns:

        A tuple containing the base name and extension.

    """

    if '.' not in file_name:

        return file_name, ''

    base_name, extension = file_name.rsplit('.', 1)

    return base_name, extension

