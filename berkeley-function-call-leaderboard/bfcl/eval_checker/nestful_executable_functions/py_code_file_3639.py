import re

from typing import Dict



def create_tag_frequency(text: str) -> Dict[str, int]:

    """

    Extracts all tags from a string input and returns a dictionary

    where the keys are the tag names and the values are their frequencies.



    Args:

        text: The input string.



    Returns:

        A dictionary where the keys are the tag names and the values are their frequencies.

    """

    tag_freq = {}

    tags = re.findall(r'<(\w+).*?>', text)

    for tag in tags:

        tag_freq[tag] = tag_freq.get(tag, 0) + 1

    return tag_freq

