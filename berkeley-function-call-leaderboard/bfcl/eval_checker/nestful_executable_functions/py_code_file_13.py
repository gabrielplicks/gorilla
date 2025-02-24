from typing import Union



def convert_string_to_float(string: str) -> Union[float, None]:

    """Converts a string to a float.



    If the string is empty, the function returns None. Otherwise, it tries to convert the string to a float and returns the result. If the conversion fails, the function returns a negative number.



    Args:

        string: The input string to be converted to a float.



    Returns:

        The converted float value if the conversion is successful, None if the string is empty, or a negative number if the conversion fails.

    """

    if not string:

        return None

    try:

        return float(string)

    except ValueError:

        return -1.0

