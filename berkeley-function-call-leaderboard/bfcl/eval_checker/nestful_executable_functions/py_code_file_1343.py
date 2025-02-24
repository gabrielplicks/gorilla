from typing import Any, Iterable



def check_iterable(obj: Any) -> None:

    """Check if an object is iterable in Python using the `__iter__` method.

    If the object is not iterable, raise a `TypeError` exception.

    Args:

        obj: The object to check if it is iterable.

    """

    try:

        iter(obj)

    except TypeError:

        raise TypeError("The object is not iterable.")

