from typing import Any



def all_attributes_set_to(obj: Any, value: Any) -> bool:

    """

    Checks if all attributes of `obj` are set to `value`.

    Returns True if all attributes are set to `value`, and False otherwise.

    Args:

        obj: The object to check.

        value: The value to check against.

    """

    for attr in vars(obj):

        if getattr(obj, attr) != value:

            return False

    return True

