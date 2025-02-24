import re

from typing import List, Tuple



def split_tags(html: str) -> List[Tuple[str, str]]:

    """Splits an HTML string into a list of tuples, where each tuple represents a tag and its corresponding content.

    The function returns an empty list if the input is not well-formed.

    For each tag, the tuple contains the tag name and its content, where the content may include nested tags.



    Args:

        html: The HTML string to be split.



    Returns:

        A list of tuples, where each tuple represents a tag and its corresponding content.

    """

    tag_pattern = r'<(\w+)>(.*?)</\1>'

    matches = re.findall(tag_pattern, html)

    if not matches:

        return []

    return [(tag, content.strip()) for tag, content in matches]

