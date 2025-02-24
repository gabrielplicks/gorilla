from typing import Union



def to_float_or_int(s: str) -> Union[float, int]:

    """Converts a string to a float or int value, depending on whether the string contains a decimal point.



    Args:

        s: The input string.



    Returns:

        The float or int value of the input string.

    """

    if '.' in s:

        return float(s)

    else:

        return int(s)

