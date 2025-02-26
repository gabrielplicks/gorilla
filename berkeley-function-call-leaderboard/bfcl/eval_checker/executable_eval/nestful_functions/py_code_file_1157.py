import re

import typing



def strip_control_characters(text: str) -> str:

    """Strips control characters from a given string.



    The control characters include the following code points:

    * U+0000 to U+001F

    * U+007F to U+009F



    Args:

        text: The input string.



    Returns:

        The input string with control characters removed.

    """

    control_characters = re.compile(r'[\x00-\x1F\x7F-\x9F]')

    return control_characters.sub('', text)

