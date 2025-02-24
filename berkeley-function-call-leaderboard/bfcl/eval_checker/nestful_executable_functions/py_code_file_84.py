from typing import Optional



def ends_with_suffix(string: str, suffix: str) -> bool:

    """Checks whether a string ends with a specified suffix.



    Args:

        string: The input string.

        suffix: The suffix to check.



    Returns:

        A boolean value indicating whether the input string ends with the specified suffix.

    """

    if not string:

        return False

    return string.endswith(suffix)

