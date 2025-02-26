from typing import Union



def to_f(input_string: str) -> Union[float, None]:

    """Converts a number in any string format to a float.



    Args:

        input_string: The input string containing a number.



    Returns:

        The input number as a float, or None if the input string is not a valid number.

    """

    try:

        return float(input_string)

    except ValueError:

        return None

