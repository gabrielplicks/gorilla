from typing import Optional



def get_last_word(string: str) -> Optional[str]:

    """

    Returns the last word in the input string.



    Args:

        string: The input string.



    Raises:

        ValueError: If the input string is empty.

    """

    if not string:

        raise ValueError('Empty string provided.')



    words = string.split()

    return words[-1]

