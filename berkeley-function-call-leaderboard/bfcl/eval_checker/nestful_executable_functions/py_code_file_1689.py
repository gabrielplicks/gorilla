from typing import Optional



def first_occurrence_index(string: str, substring: str) -> Optional[int]:

    """Calculates the index of the first occurrence of a substring within a string.

    Args:

        string: The string to search in.

        substring: The substring to search for.

    """

    for i in range(len(string)):

        if string[i] == substring[0] and string[i:i+len(substring)] == substring:

            return i

    return -1

