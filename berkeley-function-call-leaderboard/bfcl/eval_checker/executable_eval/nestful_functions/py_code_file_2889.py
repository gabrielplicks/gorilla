import re

import typing



def replace_non_printable_characters(text: str) -> str:

    """Replaces all non-printable characters with a single space and multiple spaces, tabs, and newlines with a single space.



    Args:

        text: The input text.



    Returns:

        The modified text.

    """

    text = re.sub(r'[^\x20-\x7E]+', ' ', text)

    text = re.sub(r'[\s\t\n]+', ' ', text)



    return text

