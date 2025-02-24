from typing import Union



def is_empty_string(s: Union[str, None]) -> bool:

    """Checks if the given string is empty. If the string is None, it is treated as an empty string.



    Args:

        s: The string to check.

    """

    if s is None:

        return True

    return not bool(s)

