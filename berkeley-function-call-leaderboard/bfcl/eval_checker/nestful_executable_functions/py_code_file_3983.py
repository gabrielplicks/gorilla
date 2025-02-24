from typing import Union



def format_float_as_percentage(value: Union[float, int], places: int = 2) -> str:

    """Formats a floating point value as a percentage string rounded to a specified number of decimal places.



    Args:

        value: The floating point value to be formatted as a percentage.

        places: The number of decimal places to round the value to. Default is 2.

    """

    percentage = "{:.{}%}".format(value, places)

    return percentage

