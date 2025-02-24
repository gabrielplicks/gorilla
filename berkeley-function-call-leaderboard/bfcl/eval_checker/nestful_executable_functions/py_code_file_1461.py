import os.path

from typing import Optional



def extract_filename_extension_from_path(path: str) -> Optional[tuple]:

    """Extracts the filename and extension from a given file path.



    Args:

        path: The file path.



    Returns:

        A tuple containing the filename and extension, or None if the path is empty or has no filename.

    """

    if not path:

        return

    filename, extension = os.path.splitext(os.path.split(path)[-1])

    if not filename:

        return

    return filename, extension[1:] if extension else None

