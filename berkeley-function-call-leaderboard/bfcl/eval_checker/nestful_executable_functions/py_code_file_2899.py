import os



def check_path_validity(path: str) -> bool:

    """Check if a path is valid and not a directory.



    Args:

        path: The path to check.



    Returns:

        True if the path is valid and not a directory, False otherwise.

    """

    if not os.path.isfile(path):

        return False

    try:

        with open(path, 'r') as file:

            return True

    except FileNotFoundError:

        return False

    except OSError:

        raise

