from typing import Any



def check_object_attribute(object: Any, attribute: str) -> bool:

    """Checks if a given object has a specific attribute.

    Args:

        object: The object to check.

        attribute: The attribute to check for.

    Returns:

        True if the object has the attribute, False otherwise.

    """

    try:

        return hasattr(object, attribute)

    except Exception:

        return False

