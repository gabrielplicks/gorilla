import sys



def check_python_version() -> bool:

    """Checks the current Python version and raises a ValueError if it is older than version 3.8.



    Returns:

        True if the version is 3.8 or newer.



    Raises:

        ValueError: If the Python version is older than 3.8.

    """

    if sys.version_info < (3, 8):

        raise ValueError("Python version is older than 3.8")

    return True

