from typing import Any



def is_valid_boolean(obj: Any) -> bool:

    """Checks if the object has an attribute called `is_valid` and if its value is a boolean.



    Args:

        obj: The object to check.



    Returns:

        True if the object has an attribute called `is_valid` and its value is a boolean.



    Raises:

        TypeError: If the attribute `is_valid` is not a boolean.

    """

    if hasattr(obj, 'is_valid') and isinstance(obj.is_valid, bool):

        return True

    else:

        raise TypeError('Attribute is_valid is not a boolean.')

