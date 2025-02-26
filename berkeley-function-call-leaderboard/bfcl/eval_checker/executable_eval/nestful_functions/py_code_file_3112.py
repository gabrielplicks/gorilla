import re

from typing import List



def capitalize_first_two_letters(string: str) -> str:

    """Capitalizes the first two letters of each word in a string.



    Args:

        string: The input string.



    Returns:

        A new string with the first two letters of each word capitalized.

    """

    words: List[str] = re.split(r"(\W+)", string)

    capitalized_words: List[str] = [word.title() if word.isalpha() else word for word in words]

    new_string: str = "".join(capitalized_words)

    return new_string

