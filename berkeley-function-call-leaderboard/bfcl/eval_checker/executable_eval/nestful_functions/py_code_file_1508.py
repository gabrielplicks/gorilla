import os



def get_filename_parts(filename: str) -> tuple:

    """Returns a tuple containing the filename without the extension and the extension, respectively.

    If the filename has no extension, then return the filename and an empty string for the extension.

    Args:

        filename: The filename to split into base and extension.

    """

    base, ext = os.path.splitext(filename)

    return base, ext

