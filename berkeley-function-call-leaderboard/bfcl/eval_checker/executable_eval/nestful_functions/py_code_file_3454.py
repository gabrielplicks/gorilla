from typing import Any



def is_object_iterable(obj: Any) -> bool:

    """Checks if a given object is iterable or not.



    Args:

        obj: The object to check.



    Returns:

        True if the object is iterable, False otherwise.

    """

    try:

        iter(obj)

        return True

    except TypeError:

        return False

