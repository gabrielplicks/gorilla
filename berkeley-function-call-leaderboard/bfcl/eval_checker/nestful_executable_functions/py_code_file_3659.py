import os

from typing import List



def get_file_names_with_ext(path: str, ext: str) -> List[str]:

    """Recursively traverses a directory tree starting from the given path and returns a list of file names that match the given extension.



    Args:

        path: The directory path to start traversing.

        ext: The file extension to match.



    Returns:

        A list of file names that match the given extension.

    """

    file_names = []

    try:

        for root, dirs, files in os.walk(path):

            for file in files:

                base_name, file_ext = os.path.splitext(file)

                if file_ext == '.' + ext:

                    file_names.append(os.path.join(root, file))

    except Exception:

        pass  # Handle any potential errors by returning an empty list

    return file_names

