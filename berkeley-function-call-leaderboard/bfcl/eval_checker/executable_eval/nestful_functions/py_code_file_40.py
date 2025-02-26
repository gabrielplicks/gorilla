import os



def check_file_path_prefix(file_path: str, prefix: str = None) -> bool:

    """Checks if the given file path starts with a specific prefix string.



    If the prefix string is not provided, the function returns True if the file is within the current directory (i.e., the file path starts with './').



    Args:

        file_path: The file path to check.

        prefix: The prefix string to search for.



    Returns:

        True if the file path starts with the given prefix string or if it is within the current directory, False otherwise.

    """

    if prefix is None:

        current_dir = os.getcwd()

        if file_path.startswith(current_dir):

            return True

        return False



    return file_path.startswith(prefix)

