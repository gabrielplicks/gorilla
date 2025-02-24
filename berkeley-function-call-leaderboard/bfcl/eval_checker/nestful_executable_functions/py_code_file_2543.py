import os

from typing import Dict



def construct_file_dict(directory: str) -> Dict[str, str]:

    """Constructs a dictionary of file paths and their associated names.



    The keys of the dictionary are the file paths and the values are the file names.

    The function recursively traverses a directory tree starting from the specified

    `directory` parameter.



    Args:

        directory: The root directory of the tree.



    Returns:

        A dictionary of file paths and their associated names.

    """

    file_dict = {}



    for root, dirs, files in os.walk(directory):

        for file in files:

            file_path = os.path.join(root, file)

            file_name = os.path.basename(file_path)

            file_dict[file_path] = file_name



    return file_dict

