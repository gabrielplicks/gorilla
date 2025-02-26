from typing import List



def count_numeric_words(strings: List[str]) -> int:

    """Counts the number of strings in a list that contain at least one numeric character.



    Args:

        strings: A list of strings to check for numeric characters.



    Returns:

        The number of strings in the list that contain at least one numeric character.

    """

    count = 0

    for string in strings:

        string = string.lower()

        if any(char.isnumeric() for char in string):

            count += 1

    return count

