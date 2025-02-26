from collections import Counter

from typing import Dict, List, Union



def count_chars_in_string(s1: str, s2: str) -> str:

    """

    Returns a string that represents the number of occurrences of each character from s1 in s2.

    If the character from s1 is not found in s2, the function returns a string with the character from s1 followed by 0.



    Args:

        s1: A string containing the characters to count in s2.

        s2: A string containing the characters to search for.



    Returns:

        A string with the character from s1 followed by the number of occurrences in s2.

    """

    result = ""

    counter: Dict[str, int] = Counter(s2)

    for char in s1:

        result += f"{char}{counter[char]}"

    return result

