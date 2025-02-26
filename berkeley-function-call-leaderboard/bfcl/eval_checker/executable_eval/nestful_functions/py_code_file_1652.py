import re

from typing import Optional



def replace_hostname_with_wildcard(url: str) -> Optional[str]:

    """

    Replaces the hostname in a URL with a wildcard.



    Args:

        url: The URL to modify.



    Returns:

        The modified URL with the hostname replaced with a wildcard, or None if the URL does not match the expected format.

    """

    url_pattern = r"^(https?:\/\/)([^\/]+)\/(.*)$"  # Regex pattern to match protocol, hostname, and path

    match = re.search(url_pattern, url)

    if match:

        protocol, hostname, path = match.groups()

        return f"{protocol}*example.com/{path}"

    else:

        return None

