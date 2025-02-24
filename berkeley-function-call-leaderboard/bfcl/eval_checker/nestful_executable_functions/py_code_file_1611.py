from typing import AnyStr



def remove_nonalnum(s: AnyStr) -> AnyStr:

    """Removes any non-alphanumeric characters from the input string and removes leading and trailing whitespace.



    Args:

        s: The input string.



    Returns:

        The filtered string with only alphanumeric characters and no leading or trailing whitespace.

    """

    filtered = ''.join(c for c in s if c.isalnum())

    return filtered.strip()

