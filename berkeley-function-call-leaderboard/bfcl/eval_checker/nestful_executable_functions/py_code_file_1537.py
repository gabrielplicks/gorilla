from typing import List



def split_and_trim(s: str) -> List[str]:

    """

    Splits a string into a list of non-empty, trimmed words.



    Args:

        s: The input string.



    Returns:

        A list of non-empty, trimmed words.

    """

    words = s.split(',')

    return [word.strip() for word in words if word]

