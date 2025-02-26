from typing import Union



def coerce_to_int_or_float(value: Union[int, float, str]) -> Union[int, float, None]:

    """

    Converts a value to an int or float based on the type of the input.

    If the input is not a valid type (int, float, or str), returns None.



    Args:

        value: The value to be converted.

    """

    if isinstance(value, str):

        try:

            return int(value)

        except ValueError:

            try:

                return float(value)

            except ValueError:

                return None

    elif isinstance(value, int) or isinstance(value, float):

        return value

    else:

        return None

