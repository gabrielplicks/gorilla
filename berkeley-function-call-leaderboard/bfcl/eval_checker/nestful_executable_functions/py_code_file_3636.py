from typing import Union



def convert_to_float_or_string(input_string: str) -> Union[float, int, str]:

    """Converts a string to a float if possible, or returns the original string otherwise.



    Args:

        input_string: The input string to be converted.



    Returns:

        The converted float or the original string.

    """

    try:

        converted_float = float(input_string)

        if converted_float == round(converted_float):

            return int(converted_float)

        return converted_float

    except ValueError:

        return input_string

