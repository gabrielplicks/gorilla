import re

import typing



def concatenate_identifiers(identifiers: typing.List[str], html_string: str) -> str:

    """Concatenates identifiers found in the HTML document.



    Args:

        identifiers: A list of identifier strings.

        html_string: The HTML document as a Python string.



    Returns:

        A string containing the concatenated identifiers.

    """

    matches = re.findall(r'id="(\w+)"', html_string)

    return ' '.join(matches)

