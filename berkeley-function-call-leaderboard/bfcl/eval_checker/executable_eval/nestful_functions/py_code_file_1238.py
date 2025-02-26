from typing import Any



def is_binary_view(obj: Any) -> bool:

    """Checks if a given object is a binary view.



    A binary view is an object that has a `tobytes` method and a `nbytes` attribute.



    Args:

        obj: The object to check.



    Returns:

        True if the object is a binary view and False otherwise.

    """

    return hasattr(obj, 'tobytes') and hasattr(obj, 'nbytes')

