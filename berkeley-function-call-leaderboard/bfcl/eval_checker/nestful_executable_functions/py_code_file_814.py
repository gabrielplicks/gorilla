import os.path

from typing import Tuple



def get_file_with_extension(file_name: str, extension: str) -> str:

    """Returns a new file name with a specified extension, or the original file name if the extension is already present.

    The extension should not be case-sensitive.



    Args:

        file_name: The name of the file.

        extension: The desired file extension.



    Returns:

        The file name with the specified extension.

    """

    file_name_without_extension, file_extension = os.path.splitext(file_name)

    if file_extension.lower() == extension.lower():

        return file_name

    else:

        return file_name_without_extension + extension

