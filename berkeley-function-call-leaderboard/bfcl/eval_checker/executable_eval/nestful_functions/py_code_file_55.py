from typing import Set



def check_vowels(string: str) -> bool:

    """Checks if all vowels (a, e, i, o, u, y) are present in the string.

    Args:

        string: The input string.

    Returns:

        True if all vowels are present in the string, False otherwise.

    """

    vowels: Set[str] = set(['a', 'e', 'i', 'o', 'u', 'y'])

    chars: Set[str] = set()

    for char in string.lower():

        if char.isalpha():

            chars.add(char)

    return vowels.issubset(chars)

