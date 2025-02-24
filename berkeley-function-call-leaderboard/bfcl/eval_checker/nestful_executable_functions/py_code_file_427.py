from typing import Dict



def max_len_substring_same_char(s: str) -> int:

    """Calculates the maximum length of a substring of `s` that consists of only the same character.



    Args:

        s: The input string.



    Returns:

        The maximum length of a substring of `s` that consists of only the same character.

    """

    char_counts: Dict[str, int] = {}



    max_len = 0

    for c in s:

        if c not in char_counts:

            char_counts[c] = 1

        else:

            char_counts[c] += 1

        max_len = max(max_len, char_counts[c])



    return max_len

