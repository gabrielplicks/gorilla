import os



def get_parent_dir_path(file_path: str) -> str:

    """Returns the parent directory path of a given file path.



    Args:

        file_path: The path to the file.

    """

    parent_dir_path = os.path.dirname(file_path)

    return parent_dir_path

