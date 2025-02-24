import os



def extract_stem_filename(path: str) -> str:

    """Extracts the stem filename from a path string.



    The stem filename is the filename without the extension.



    Args:

        path: The path string.



    Returns:

        The stem filename.

    """

    head, tail = os.path.split(path)

    filename, extension = os.path.splitext(tail)

    return filename

