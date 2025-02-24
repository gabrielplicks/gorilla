from typing import List



def find_string_with_repeated_characters(arr: List[str]) -> str:

    """Finds the first (lexicographically) string that has at least one character that appears at least twice.



    Args:

        arr: A list of strings.



    Returns:

        The first (lexicographically) string that has at least one character that appears at least twice.

        If there is no such string, return an empty string.

    """

    for s in arr:

        seen = set()

        for c in s:

            if c in seen:

                return s

            seen.add(c)

    return ""

