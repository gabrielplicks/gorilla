import os

from typing import Tuple



def get_file_type_and_name(file_path: str) -> Tuple[str, str]:

    """Extracts the file type and name from a file path.



    Args:

        file_path: A string representing the file path.



    Returns:

        A tuple of strings containing the file type and file name.

    """

    file_name, file_ext = os.path.splitext(file_path)

    file_type = file_ext[1:]  # Remove the leading dot from the extension

    file_name = file_name.split('/')[-1]

    file_type = file_type if file_type else ''

    return (file_type, file_name)

