import glob

import os

from typing import List



def find_files_containing_example(path: str) -> List[str]:

    """

    Finds files in the given path that contain the word 'example' in their names.



    Args:

        path: The path to search for files.



    Returns:

        A list of file names that contain the word 'example'.

    """

    file_names = []



    for file_path in glob.glob(os.path.join(path, '*example*')):

        file_name, file_ext = os.path.splitext(os.path.basename(file_path))

        if 'example' in file_name:

            file_names.append(file_name)



    return file_names

