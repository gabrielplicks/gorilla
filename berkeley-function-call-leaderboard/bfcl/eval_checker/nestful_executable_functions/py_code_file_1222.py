from typing import Any



def check_attr(obj: Any, attr: str, value: Any) -> bool:

    """Checks if an object has a given attribute and whether its value is equal to a given value.



    Args:

        obj: The object to check.

        attr: The name of the attribute to check.

        value: The value to compare the attribute's value to.

    """

    if hasattr(obj, attr):

        return getattr(obj, attr) == value

    else:

        return False

