from typing import Union



def to_str_if_number_or_string(obj: Union[int, float, str]) -> Union[str, None]:

    """Converts a number or string to a string, and returns None for other types.



    Args:

        obj: The object to convert.



    Returns:

        The object as a string, or None if the object is not a number or string.

    """

    if isinstance(obj, (int, float, str)):

        return str(obj)

    else:

        return None

