import os



def extract_file_extension(path: str) -> str:

    """Extracts the file name extension from a file path.



    Args:

        path: The file path.



    Returns:

        The file name extension as a string, without the leading period.

        If the path does not have an extension, an empty string is returned.

    """

    base_name, extension = os.path.splitext(path)

    return extension[1:] if extension else ''

