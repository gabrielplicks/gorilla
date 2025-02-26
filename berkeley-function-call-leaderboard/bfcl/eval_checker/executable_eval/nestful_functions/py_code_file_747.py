from typing import Union



def number_of_words(string: Union[str, None]) -> int:

    """Returns the number of words in a given string. If the input string is None, returns 0.



    Args:

        string: The input string.

    """

    if not string:

        return 0

    return len(string.split())

