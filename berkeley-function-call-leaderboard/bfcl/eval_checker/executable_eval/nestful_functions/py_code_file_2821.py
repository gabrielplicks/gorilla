import re



def is_https(url: str) -> bool:

    """Determines whether a given URL is using the HTTPS protocol.



    Args:

        url: The URL to check.



    Returns:

        True if the URL is using HTTPS, False otherwise.

    """

    scheme_match = re.search(r'^(https?|ftp)://', url)

    if scheme_match:

        return scheme_match.group(1) == 'https'

    else:

        return False

