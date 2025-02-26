import re

import typing



def remove_non_digits_lowercase_and_dash(string: str) -> str:

    """Removes all characters from a string except digits, lowercase letters, and the dash character ('-').



    Args:

        string: The input string.



    Returns:

        The modified string.

    """

    return re.sub("[^0-9a-z-]", "", string, flags=re.IGNORECASE)

