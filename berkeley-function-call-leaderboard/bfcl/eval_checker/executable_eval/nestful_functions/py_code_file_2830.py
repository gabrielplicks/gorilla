from typing import Any



def is_not_zero(obj: Any) -> bool:

    """Checks if an object has a `value` attribute and that its value is not zero.

    If the object does not have a `value` attribute, the function returns True.



    Args:

        obj: The object to check.



    Returns:

        True if the object does not have a `value` attribute or its value is not zero.

        False if the object has a `value` attribute and its value is zero.

    """

    if hasattr(obj, "value"):

        if obj.value == 0:

            return False

        else:

            return True

    else:

        return True

