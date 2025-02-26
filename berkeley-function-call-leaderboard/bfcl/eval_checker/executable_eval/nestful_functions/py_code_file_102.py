from typing import Dict



def most_frequent_character(string: str) -> str:

    """Returns the character that appears the most frequently in the string, ignoring differences in case.

    If two characters have the same number of occurrences, the function returns the one that appears earlier in the string.



    Args:

        string: The input string.



    Returns:

        The most frequent character in the string.

    """

    char_count: Dict[str, int] = {}



    for char in string.lower():

        char_count[char] = char_count.get(char, 0) + 1



    max_count = max(char_count.values())

    most_frequent_chars = [char for char, count in char_count.items() if count == max_count]



    return most_frequent_chars[0]

