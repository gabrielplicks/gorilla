import re

from typing import List, Tuple



def extract_hostnames_from_urls(urls: List[str]) -> List[Tuple[str, str]]:

    """Extracts the hostnames from a list of URLs.



    Args:

        urls: A list of URLs.



    Returns:

        A list of tuples, where each tuple contains the original URL and the extracted hostname.

    """

    hostnames = []

    for url in urls:

        try:

            match = re.search(r'//([\w.-]*)', url)

            if match:

                hostname = match.group(1)

                hostnames.append((url, hostname))

            else:

                hostnames.append((url, None))

        except Exception:

            hostnames.append((url, None))



    return hostnames

