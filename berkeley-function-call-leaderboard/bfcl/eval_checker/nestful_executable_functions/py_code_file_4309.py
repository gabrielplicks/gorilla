from typing import List



def split_int_to_digits(input_str: str) -> List[int]:

    """Splits a positive integer (represented as a string) into individual digits.



    Args:

        input_str: The input string.



    Raises:

        ValueError: If the input string is not a valid integer.

    """

    if not input_str.isdigit():

        raise ValueError("Invalid integer.")



    return [int(c) for c in input_str]

