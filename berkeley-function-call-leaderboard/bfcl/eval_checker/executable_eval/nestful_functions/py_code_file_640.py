import os

from typing import Union



def get_file_name_from_uri(uri: Union[str, bytes]) -> str:

    """Extracts the file name from a URI, regardless of the number of directories present in the URI.

    The function returns the file name without the extension.

    Args:

        uri: The URI to a file.

    """

    parts = uri.split('/')

    file_name_with_extension = parts[-1]

    file_name, extension = os.path.splitext(file_name_with_extension)

    return file_name

