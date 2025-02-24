from typing import Union



def string_to_type(value: str, typ: str) -> Union[None, int, float, str, bool]:

    """Converts a string to a specified data type.



    Args:

        value: The string to be converted.

        typ: The data type to convert the string to.



    Returns:

        The converted value, or None if the conversion fails.

    """

    try:

        return eval(typ)(value)

    except:

        return None

