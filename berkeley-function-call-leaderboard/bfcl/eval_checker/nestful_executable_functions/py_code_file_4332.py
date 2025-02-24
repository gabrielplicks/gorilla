import os



def get_fullpath(base_path: str, file_name: str, add_trailing_slash: bool) -> str:

    """

    Returns the full path of a file given its base path and file name.

    Args:

        base_path: The base path of the file.

        file_name: The name of the file.

        add_trailing_slash: Whether to add a trailing slash to the base path if it is missing.

    """

    if not base_path.endswith(os.sep):

        if add_trailing_slash:

            base_path = base_path + os.sep



    full_path = os.path.join(base_path, file_name)

    return full_path

