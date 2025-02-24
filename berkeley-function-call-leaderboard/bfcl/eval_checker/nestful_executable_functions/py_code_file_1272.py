from typing import Any



def is_string_like(obj: Any) -> bool:

    """Checks whether an object is string-like.



    Args:

        obj: The object to check.



    Returns:

        True if `obj` is a string or any object that implements the `__str__()` method. Otherwise, False.

    """

    return isinstance(obj, str) or hasattr(obj, '__str__')

