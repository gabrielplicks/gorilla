import re

from typing import Optional



def extract_ip_address(header_string: str) -> Optional[str]:

    """Extracts the IP address from an HTTP request header string.



    Args:

        header_string: The request header string.



    Returns:

        The IP address as a string, or '?' if the IP address is unknown.

    """

    lines = header_string.split('\n')

    for line in lines:

        if line.startswith('X-Forwarded-For:'):

            ip_address = re.search(r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})', line).group(1)

            return ip_address

    return '?'

