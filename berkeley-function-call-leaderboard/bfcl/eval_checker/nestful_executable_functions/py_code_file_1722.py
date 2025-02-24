import re

from typing import List



def extract_domain_names(urls: List[str]) -> List[str]:

    """Extracts domain names from a list of URLs.



    Args:

        urls: A list of URLs.



    Returns:

        A list of domain names.

    """

    domain_names = []

    for url in urls:

        domain_name_match = re.findall(r'https?://(?:www\.)?([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)', url)

        if domain_name_match:

            domain_name = domain_name_match[0]

            domain_names.append(domain_name)

    return domain_names

