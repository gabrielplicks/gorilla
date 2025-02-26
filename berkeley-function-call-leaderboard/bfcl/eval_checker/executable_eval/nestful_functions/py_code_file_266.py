from typing import Optional



def remove_first_and_last_char(s: str) -> Optional[str]:

    """Removes the first and last characters of a string if it has at least three characters.



    Args:

        s: The input string.



    Returns:

        The string obtained by removing the first and last characters of `s` if `s` has at least three characters.

        If `s` has less than three characters, returns an empty string.

    """

    if len(s) < 3:

        return ""

    return s[1:-1]

