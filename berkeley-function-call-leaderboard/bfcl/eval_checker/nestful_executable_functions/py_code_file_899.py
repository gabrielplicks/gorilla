from typing import Union



def url_path_to_file_path(url_path: str) -> str:

    """Converts a URL path to a file path.



    Args:

        url_path: The URL path to convert.



    Returns:

        The corresponding file path.

    """

    if url_path.endswith('/'):

        return url_path + 'index.html'

    else:

        return url_path + '.html'

