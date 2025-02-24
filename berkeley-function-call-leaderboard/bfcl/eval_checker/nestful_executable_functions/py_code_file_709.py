from typing import List



def split_string_into_digits(string: str) -> List[int]:

    """Splits a string into a list of digits, where each element of the list represents an integer in the range [0, 9].



    Args:

        string: The input string.



    Returns:

        A list of integers.

    """

    if len(string) == 1:

        if string.isdigit():

            return [int(string)]

        else:

            return [ord(string) - ord('0')]



    if string[0].isdigit():

        return [int(string[0])] + split_string_into_digits(string[1:])

    else:

        return [ord(string[0]) - ord('0')] + split_string_into_digits(string[1:])

