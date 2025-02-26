from typing import Set



def unique_substring(s: str) -> str:

    """

    Finds the longest unique substring of a given string `s`.



    Args:

        s: The input string.



    Returns:

        The longest unique substring of `s`.

    """

    start = end = 0

    longest_substring = ""

    seen: Set[str] = set()



    while end < len(s):

        char = s[end]

        if char in seen:

            seen.remove(s[start])

            start += 1

        else:

            seen.add(char)

            end += 1

            if end - start > len(longest_substring):

                longest_substring = s[start:end]



    return longest_substring

