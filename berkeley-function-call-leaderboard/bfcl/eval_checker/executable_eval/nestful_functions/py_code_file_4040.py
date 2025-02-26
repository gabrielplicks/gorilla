import os



def get_directory_names(file_path: str) -> list:

    """Returns a list of directory names from the root directory to the current directory.



    Args:

        file_path: The file path string.



    Returns:

        A list of directory names.

    """

    if not file_path or file_path == os.path.sep:

        return []

    parent = os.path.dirname(file_path)

    current_dir = os.path.basename(file_path)



    return get_directory_names(parent) + [current_dir]

