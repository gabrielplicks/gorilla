from typing import Optional



def get_extension(filename: str) -> Optional[str]:

    """

    Returns the file extension of a given filename.

    If the filename has no extension, returns an empty string.



    Args:

        filename: The filename to extract the extension from.



    Returns:

        The file extension as a string, or an empty string if the filename has no extension.

    """

    dot_index = filename.rfind(".")

    if dot_index == -1:

        return ""

    return filename.split(".")[-1]

