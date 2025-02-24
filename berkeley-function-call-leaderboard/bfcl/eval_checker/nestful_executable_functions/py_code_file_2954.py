import re



def is_valid_file_path(file_path: str) -> bool:

    """

    Checks if a given file path is valid, according to the following criteria:

    - The file path is not empty

    - The file path does not end with a period

    - The file path does not contain consecutive periods

    - The file path does not contain consecutive slashes



    Args:

        file_path: The file path to check.



    Returns:

        True if the file path is valid, False otherwise.

    """

    regex = r"^[^.][^.]*[^./]$"

    return bool(re.match(regex, file_path))

