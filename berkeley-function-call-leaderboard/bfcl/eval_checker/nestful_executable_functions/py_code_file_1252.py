from typing import List



def split_into_two(string: str) -> List[str]:

    """Splits a string into two parts.



    If the length of the string is even, the length of the first part is equal to the length of the second part.

    If the length of the string is odd, the length of the first part is one greater than the length of the second part.



    Args:

        string: The string to split.



    Returns:

        A list of two strings representing the split string.

    """

    middle = len(string) // 2

    if len(string) % 2 == 0:

        return [string[:middle], string[middle:]]

    else:

        return [string[:middle + 1], string[middle + 1:]]

