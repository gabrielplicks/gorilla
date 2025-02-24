from typing import Any



def check_has_attribute(obj: Any, attr: str) -> bool:

    """Checks if the given object has the specified attribute.



    Args:

        obj: The object to check.

        attr: The attribute to check for.

    """

    return hasattr(obj, attr)

