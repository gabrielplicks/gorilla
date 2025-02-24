from typing import List



def cyclic_shifts(string: str) -> List[str]:

    """Generates a list of cyclic shifts of a given string.



    A cyclic shift is an operation that shifts the characters of a string by some amount.

    For example, given the string "abc", the cyclic shifts are "abc", "bca", and "cab".



    Args:

        string: The input string.



    Returns:

        A list of cyclic shifts of the input string.

    """

    shifts = []

    for i in range(len(string)):

        shifted_string = string[i:] + string[:i]

        shifts.append(shifted_string)

    return shifts

