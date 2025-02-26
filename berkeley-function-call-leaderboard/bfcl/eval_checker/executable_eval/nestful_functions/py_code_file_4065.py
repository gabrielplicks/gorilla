import os



def get_file_basename_without_extension(path: str) -> str:

    """Returns the base filename without the extension.

    Args:

        path: The file path.

    Raises:

        ValueError: If the file path does not exist.

    """

    if not os.path.exists(path):

        raise ValueError(f"File path '{path}' does not exist.")

    filename = os.path.basename(path)

    name, ext = os.path.splitext(filename)

    return name

