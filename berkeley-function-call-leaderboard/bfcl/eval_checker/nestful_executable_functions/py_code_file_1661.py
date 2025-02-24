from typing import Any



def check_for_attribute(obj: Any, attr_name: str) -> bool:

    """Checks if a given attribute exists on an object.



    Args:

        obj: The object to check for the attribute.

        attr_name: The name of the attribute to check for.



    Returns:

        True if the attribute exists on the object, False otherwise.

    """

    return hasattr(obj, attr_name)

