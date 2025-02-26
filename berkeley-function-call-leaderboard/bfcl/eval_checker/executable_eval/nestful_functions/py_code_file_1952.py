import tempfile



def create_temporary_directory() -> str:

    """Creates a temporary directory with a random name to hold a set of files.

    Returns:

        The path to the directory.

    """

    temp_dir = tempfile.mkdtemp()

    return temp_dir

