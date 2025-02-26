from typing import Union



def numeric_value(input_string: str) -> Union[int, float, None]:

    """Converts a string to its corresponding numeric value, if possible.

    Args:

        input_string: The input string to be converted.

    Returns:

        The numeric value of the input string, if it can be converted to an integer or a float.

        None, if the input string cannot be converted to a numeric value.

    """

    try:

        value = int(input_string)

    except ValueError:

        try:

            value = float(input_string)

        except ValueError:

            print("Invalid input, cannot convert to a numeric value")

            return None

    if type(value) in (int, float):

        return value

    else:

        print("Invalid input, expected an integer or a float")

        return None

