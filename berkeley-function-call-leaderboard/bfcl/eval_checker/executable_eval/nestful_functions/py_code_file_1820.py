from typing import Any



def has_static_member(obj: Any) -> bool:

    """Checks whether an object has a static member based on a given condition.



    Args:

        obj: The object to check.



    Returns:

        True if the object has the specified attribute or method, False otherwise.

    """

    return hasattr(obj, 'value')

