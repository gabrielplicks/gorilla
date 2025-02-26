from typing import Union



class NotAStringIntegerOrFloatError(Exception):

    """Custom exception for when an object is not a string, integer, or float."""

    pass



def check_string_integer_or_float(obj: Union[str, int, float]) -> str:

    """Checks if a given object is a string, integer, or float, and returns the type as a string.



    Args:

        obj: The object to check.



    Raises:

        NotAStringIntegerOrFloatError: If the object is not a string, integer, or float.

    """

    if isinstance(obj, str):

        return "string"

    elif isinstance(obj, int):

        return "integer"

    elif isinstance(obj, float):

        return "float"

    else:

        raise NotAStringIntegerOrFloatError("The object is not a string, integer, or float.")

