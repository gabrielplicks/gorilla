from typing import Union



def convert_number_string(input: Union[str, int]) -> Union[str, int]:

    """Converts a string to a number or a number to a string.

    Args:

        input: A string or a number.

    Raises:

        ValueError: If the input is neither a string nor a number.

    """

    if type(input) == str:

        return int(input)

    elif type(input) == int:

        return str(input)

    else:

        raise ValueError("Input must be a string or a number.")

