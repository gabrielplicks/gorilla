from typing import Union



def is_valid_boolean_string(s: Union[str, bool]) -> bool:

    """Checks if a given string is a valid boolean string.



    A boolean string is a string that is either `"True"` or `"False"`.



    Args:

        s: The string to check.



    Returns:

        `True` for a valid boolean string and `False` for an invalid string.

    """

    return s == "True" or s == "False"

