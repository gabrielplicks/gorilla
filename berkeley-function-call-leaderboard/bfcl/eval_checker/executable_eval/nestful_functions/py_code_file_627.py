import os



def get_root_path(file_path: str) -> str:

    """Obtains the current project's root path based on the file's directory location.



    Args:

        file_path: The path to the file for which we want to find the root path.



    Returns:

        The root path as a string.

    """

    dir_path = os.path.dirname(file_path)

    parts = os.path.split(dir_path)

    project_path = os.path.join(*parts[:-1])

    return project_path

