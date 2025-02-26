from typing import List



def get_subdirectories(path_string: str) -> List[str]:

    """Extracts the subdirectories of a parent directory from a string.



    Args:

        path_string: A string formatted as follows:

            parent_dir_name/sub_dir1_name/sub_dir2_name/.../sub_dirn_name



    Returns:

        A list of subdirectories.

    """

    subdirectories = path_string.split("/")

    subdirectories.pop(0)  # Remove the parent directory

    return subdirectories

