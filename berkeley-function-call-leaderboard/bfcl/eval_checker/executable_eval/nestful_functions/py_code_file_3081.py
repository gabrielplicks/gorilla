from typing import Union



def get_file_name(full_path: Union[str, bytes]) -> str:

    """Returns the file name from a full file path.



    Args:

        full_path: The full file path.

    """

    path_parts = full_path.split('/')

    return path_parts[-1]

