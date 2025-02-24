from typing import List



def split_by_unicode_whitespace(text: str) -> List[str]:

    """Splits a string into a list of tokens based on Unicode whitespace characters.



    Args:

        text: The input string to be split.



    Returns:

        A list of tokens split by Unicode whitespace characters.

    """

    return text.split()

