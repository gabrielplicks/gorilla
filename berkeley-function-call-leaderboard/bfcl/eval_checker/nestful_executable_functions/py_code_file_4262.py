from typing import Any



def count_distinct_characters(string: str) -> int:

    """Calculates the number of distinct characters in a string.



    Args:

        string: The input string.



    Raises:

        TypeError: If the input is not a string.

    """

    if not isinstance(string, str):

        raise TypeError("Input must be a string")



    distinct_chars = set()



    for char in string:

        distinct_chars.add(char)



    return len(distinct_chars)

