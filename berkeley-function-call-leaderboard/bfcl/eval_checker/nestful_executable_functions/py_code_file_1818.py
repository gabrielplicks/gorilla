from typing import Optional



def count_from_end(string: str, substring: str) -> int:

    """Counts the number of times a substring occurs in a string from the end.



    Args:

        string: The string to search in.

        substring: The substring to search for.



    Returns:

        The number of times the substring occurs in the string from the end.

    """

    if not string or len(substring) > len(string):

        return 0

    count = 0

    for i in range(len(string) - len(substring) + 1):

        if string[i:i + len(substring)] == substring:

            count += 1



    return count

