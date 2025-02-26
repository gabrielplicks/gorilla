import os



def find_from_root(root_path: str, file_name: str) -> str:

    """Scans the entire file system starting from the root path for a file with a given name.

    If found, returns the full path to the file. If not found, returns None.

    Args:

        root_path: The root path to start the search from.

        file_name: The name of the file to search for.

    """

    for current_path, directories, files in os.walk(root_path):

        if file_name in files:

            return os.path.join(current_path, file_name)

    return None

