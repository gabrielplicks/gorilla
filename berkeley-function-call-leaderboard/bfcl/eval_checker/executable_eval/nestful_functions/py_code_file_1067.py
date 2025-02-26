from typing import Dict



def is_any_char_repeated(s: str) -> bool:

    """Checks if any character in the string occurs more than once.



    Args:

        s: The input string.



    Returns:

        True if any character occurs more than once, False otherwise.

    """

    char_count: Dict[str, int] = {}

    for char in s:

        if char in char_count:

            char_count[char] += 1

        else:

            char_count[char] = 1

    return any(count > 1 for count in char_count.values())

