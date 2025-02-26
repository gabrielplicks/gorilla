import re

from typing import List



def find_url_domains(urls: List[str]) -> List[str]:

    """Extracts the domains from a list of URLs.



    If a URL has an IP address or a local domain, then extract just the host name,

    otherwise extract the subdomain and domain.



    Args:

        urls: A list of URLs.



    Returns:

        A list of domains extracted from the input URLs.

    """

    pattern = r'^(?:https?://)?(?:www\.)?(?P<domain>[^/]+)'

    return [re.match(pattern, url).group('domain') for url in urls]

