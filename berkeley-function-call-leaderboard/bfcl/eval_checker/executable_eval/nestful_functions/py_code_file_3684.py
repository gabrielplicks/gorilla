from typing import Set



def duplicate_substring(string: str) -> str:

    """Finds the longest substring that occurs at least twice in the input string.



    Args:

        string: The input string.



    Returns:

        The longest substring that occurs at least twice in the input string.

    """

    substrings: Set[str] = set()

    longest_substring: str = ""



    for i in range(len(string) - 1):

        for j in range(i + 1, len(string)):

            substring: str = string[i:j]



            if substring in string[j:] and substring not in substrings:

                substrings.add(substring)



                if len(substring) > len(longest_substring):

                    longest_substring = substring



    return longest_substring

