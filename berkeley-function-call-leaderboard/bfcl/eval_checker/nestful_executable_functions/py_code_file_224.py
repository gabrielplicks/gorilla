import os

from typing import Union



def full_path(directory: Union[str, os.PathLike], file_name: Union[str, os.PathLike]) -> str:

    """Constructs and returns the full path of the file by joining `directory` and `file_name` with a separator.

    If the file does not exist, the function raises an `OSError` with a message "File does not exist".



    Args:

        directory: The directory where the file is located.

        file_name: The name of the file.



    Raises:

        OSError: If the file does not exist.

    """

    path = os.path.join(directory, file_name)

    if not os.path.exists(path):

        raise OSError("File does not exist")



    return path

