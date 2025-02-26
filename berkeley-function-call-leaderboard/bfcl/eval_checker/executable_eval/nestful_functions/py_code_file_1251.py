import os



def search_file(dir_path: str, target_file: str) -> str:

    """Searches for a target file in a directory and its subdirectories.



    Args:

        dir_path: The directory path to search in.

        target_file: The name of the target file to search for.



    Returns:

        The full path to the target file if found, and None if the file is not found.

    """

    for dir_path, dirs, files in os.walk(dir_path):

        for file in files:

            if file == target_file:

                return os.path.join(dir_path, file)

    return None

