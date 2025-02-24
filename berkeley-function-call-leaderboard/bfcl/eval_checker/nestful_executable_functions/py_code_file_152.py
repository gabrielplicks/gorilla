import re

from typing import List



def get_links(text: str) -> List[str]:

    """Parses a string of text and returns a list of URLs, sorted by their first occurrence in the text.



    Args:

        text: The string of text to parse.



    Returns:

        A list of URLs sorted by their first occurrence in the text.

    """

    urls = re.findall(r'https?://\S+', text)

    sorted_urls = sorted(urls, key=lambda url: text.index(url))



    return sorted_urls

