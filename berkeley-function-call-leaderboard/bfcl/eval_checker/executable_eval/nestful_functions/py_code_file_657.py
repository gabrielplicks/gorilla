from typing import List



def non_whitespace_substrings(input_str: str) -> List[str]:

    """

    Returns a list of non-empty substrings that do not contain any whitespace characters.

    Args:

        input_str: A string containing multiple words separated by whitespace characters.

    """

    words = input_str.strip().split()

    return [word for word in words if word and not word.isspace()]

