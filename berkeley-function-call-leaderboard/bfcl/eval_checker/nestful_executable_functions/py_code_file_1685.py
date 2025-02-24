import os



def is_file_accessible(path: str) -> bool:

    """Checks if a file is accessible at a given path.



    Args:

        path: The path to the file.



    Returns:

        A boolean indicating whether the file is accessible.

    """

    try:

        return os.access(path, os.R_OK)

    except Exception as e:

        print(f'Error: {e}')

        return False

