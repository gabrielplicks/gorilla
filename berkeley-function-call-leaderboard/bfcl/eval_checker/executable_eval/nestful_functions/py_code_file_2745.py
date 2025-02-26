from typing import List



def find_longest_non_reversed(words: List[str]) -> int:

    """Finds the length of the longest string in the list where the string is not the reverse of another string in the list.



    Args:

        words: A list of strings.



    Returns:

        The length of the longest string in the list where the string is not the reverse of another string in the list.

    """

    longest = 0

    for word in words:

        if word != word[::-1]:

            longest = max(longest, len(word))

    return longest

