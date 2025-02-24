import re

from typing import List



def collect_tags(html_elements: List[str]) -> List[str]:

    """Collects the tags from a list of HTML elements.



    Args:

        html_elements: A list of HTML elements represented as strings.



    Returns:

        A list of the tags contained in each element as strings.

    """

    tags = []

    for element in html_elements:

        tag_match = re.search(r"<(\w+)", element)

        if tag_match:

            tags.append(tag_match.group(1))

    return tags

