from typing import Union



def convert_val_to_float(val: Union[str, int, float, None]) -> Union[float, None]:

    """Converts a string, integer, or float to a floating-point value.



    Args:

        val: The value to convert.



    Returns:

        The converted value as a floating-point value, or None if the conversion is not possible.

    """

    if isinstance(val, str):

        try:

            return float(val)

        except ValueError:

            return None

    elif isinstance(val, int):

        return float(val)

    elif isinstance(val, float):

        return val

    elif val is None:

        return 0.0

    else:

        return None

