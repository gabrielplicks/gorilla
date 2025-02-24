from typing import Any



def attribute_exists_and_is_not_none(obj: Any, attribute: str) -> bool:

    """Checks if an object has a specified attribute and if it is not None.



    Args:

        obj: The object to check.

        attribute: The attribute to check.



    Returns:

        True if the attribute exists and is not None, False otherwise.

    """

    if hasattr(obj, attribute) and getattr(obj, attribute) is not None:

        return True

    return False

