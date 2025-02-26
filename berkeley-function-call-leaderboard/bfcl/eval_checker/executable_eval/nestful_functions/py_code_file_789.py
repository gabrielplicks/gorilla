from typing import List



def split_string_by_commas(input_string: str) -> List[int]:

    """Splits a given string by commas and returns a list of integers.

    Args:

        input_string: The input string to be split.

    Raises:

        AssertionError: If the input is not a string.

        ValueError: If the input string cannot be split by commas.

    """

    assert isinstance(input_string, str), "Input must be a string"

    if "," not in input_string:

        raise ValueError("Input string must contain at least one comma")

    return [int(x) for x in input_string.split(',')]

