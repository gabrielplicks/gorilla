from typing import Dict



def count_characters(s: str) -> str:

    """Counts the occurrences of each character in a string and returns a string with the format `{char}={count}`.



    Args:

        s: The input string.



    Returns:

        A string containing the number of occurrences of each character in `s`.

    """

    char_counts: Dict[str, int] = {}

    for char in s:

        if char in char_counts:

            char_counts[char] += 1

        else:

            char_counts[char] = 1

    return ''.join([f'{char}={count}' for char, count in sorted(char_counts.items())])

