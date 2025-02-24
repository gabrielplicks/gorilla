from typing import Dict



def longest_substring_no_duplicates(s: str) -> int:

    """Calculates the length of the longest substring that contains no duplicate characters.



    Args:

        s: The input string.



    Returns:

        The length of the longest substring that contains no duplicate characters.

    """

    start, longest = 0, 0

    seen: Dict[str, int] = {}

    for end in range(len(s)):

        if s[end] in seen:

            start = max(start, seen[s[end]] + 1)

        seen[s[end]] = end

        longest = max(longest, end - start + 1)

    return longest

