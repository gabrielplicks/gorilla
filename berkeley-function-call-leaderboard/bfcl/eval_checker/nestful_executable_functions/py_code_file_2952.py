import os



def has_json_extension(file_path: str) -> bool:

    """Checks if a given file path has the ".json" extension.



    Args:

        file_path: The file path to check.



    Returns:

        True if the file path has the ".json" extension and False otherwise.

    """

    _, extension = os.path.splitext(file_path)

    return extension.lower() == ".json"

