from typing import Optional



def replace_letters(s: str) -> Optional[str]:

    """Replaces all occurrences of 'A' with '4', 'E' with '3', and 'G' with '6' in a given string.



    Args:

        s: The input string.



    Returns:

        The modified string, or None if the input is not a string.

    """

    if not isinstance(s, str):

        return None

    return s.replace("A", "4").replace("E", "3").replace("G", "6")

