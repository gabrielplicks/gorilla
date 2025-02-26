import urllib.request

import urllib.error



def check_url_status(url: str) -> bool:

    """Checks if a URL returns a valid HTTP status code.



    Args:

        url: The URL to check.



    Returns:

        True if the URL returns a valid HTTP status code, False otherwise.

    """

    try:

        response = urllib.request.urlopen(url)

        return True

    except urllib.error.URLError:

        return False

