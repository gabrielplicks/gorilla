import re

from typing import List



def find_all_hrefs(html_code: str) -> List[str]:

    """Finds all URLs of hyperlinks <a> tags in the given HTML code.



    Args:

        html_code: The HTML code to search for <a> tags.



    Returns:

        A list of strings representing the URLs of the <a> tags.

    """

    href_pattern = re.compile(r'<a\s+href=["\']([^"\'\s>]+)["\']', re.IGNORECASE)

    hrefs = href_pattern.findall(html_code)

    return hrefs

