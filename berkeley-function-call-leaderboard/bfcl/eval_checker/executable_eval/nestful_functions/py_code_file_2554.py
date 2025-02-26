import re

import typing



def remove_non_alphanumeric_characters(s: str) -> str:

    """Removes all characters from the string that are not alphanumeric.

    Args:

        s: The input string.

    """

    return re.sub(r'[\W_]+', '', s)

