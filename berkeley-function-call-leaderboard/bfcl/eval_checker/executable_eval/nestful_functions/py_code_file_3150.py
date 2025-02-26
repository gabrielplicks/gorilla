from typing import Set



def longest_substring_length(s: str) -> int:

    """Calculates the length of the longest substring with no repeated characters in a given string.



    Args:

        s: The input string.



    Returns:

        The length of the longest substring with no repeated characters.

    """

    start: int = 0

    end: int = 0

    longest_length: int = 0

    seen: Set[str] = set()



    while end < len(s):

        if s[end] not in seen:

            seen.add(s[end])

            end += 1

            longest_length = max(longest_length, end - start)

        else:

            seen.remove(s[start])

            start += 1



    return longest_length

