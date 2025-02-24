import os



def get_extension_from_path(file_path: str) -> str:

    """Extracts the file extension from a given file path.



    Args:

        file_path: The file path to extract the extension from.



    Returns:

        The file extension after the last dot (`.`) in the path, or an empty string if the path does not contain a dot.



    Raises:

        FileNotFoundError: If the given file path is invalid.

    """

    root, extension = os.path.splitext(file_path)

    if not root:  # If the given file path is invalid

        print("Error: Invalid file path.")

        raise FileNotFoundError

    return extension

