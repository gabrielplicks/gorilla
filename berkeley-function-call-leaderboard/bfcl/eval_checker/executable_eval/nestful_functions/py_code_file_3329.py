from typing import Union



def string_to_int_or_float(s: str) -> Union[int, float]:

    """Converts a string to an integer or a float.



    Args:

        s: The string to convert.



    Returns:

        The integer or float representation of the string.

    """

    return eval(s)

