from typing import Union

import os

import pathlib



def get_longest_extension_file(dir_path: Union[str, pathlib.Path]) -> str:

    """Returns the longest filename extension from all files in the directory.

    Args:

        dir_path: The directory path.

    """

    dir_path = pathlib.Path(dir_path).resolve()  # Resolve relative paths

    file_name_length = 0

    longest_extension_file = ""

    for file_name in os.listdir(dir_path):

        if len(file_name) > file_name_length:

            file_name_length = len(file_name)

            longest_extension_file = file_name

    return longest_extension_file

