from typing import List



def is_file_path_valid(file_path: str, extension: str) -> bool:

    """Checks if a file path contains a given extension.



    Args:

        file_path: The file path to check.

        extension: The file extension to check for.



    Returns:

        True if the file path contains the extension, False otherwise.

    """

    if not file_path:

        return False

    if not extension:

        return True

    if any(file_path.endswith(f".{ext}") for ext in extension.split(',')):

        return True

    return False

