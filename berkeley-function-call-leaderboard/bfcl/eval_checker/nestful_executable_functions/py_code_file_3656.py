from typing import List, Union



def convert_to_digits(digit_string: Union[str, List[int]]) -> List[int]:

    """Converts a digit string to a list of digits.

    Args:

        digit_string: A string containing only digits (from 0 to 9) or a list of digits.

    """

    if isinstance(digit_string, str):

        return list(map(int, digit_string))

    elif isinstance(digit_string, list):

        return digit_string

    else:

        raise ValueError("Invalid input type. Expected a string or list of digits.")

