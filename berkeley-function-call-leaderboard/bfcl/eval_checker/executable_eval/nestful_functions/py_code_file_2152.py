from typing import Optional



def bool_or_none(value: bool) -> Optional[bool]:

    """

    Return the boolean value of the argument if it is a boolean, or None if it is not a boolean.



    Args:

        value: The input value to check.

    """

    if isinstance(value, bool):

        return value

    else:

        return None

