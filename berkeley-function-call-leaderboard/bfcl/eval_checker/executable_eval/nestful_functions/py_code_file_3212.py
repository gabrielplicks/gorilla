from typing import Union



def remove_trailing_slash(url: Union[str, bytes]) -> Union[str, bytes]:

    """Removes the trailing slash from a URL.



    Args:

        url: The URL to clean.



    Returns:

        The cleaned URL.

    """

    return url.rstrip('/')

