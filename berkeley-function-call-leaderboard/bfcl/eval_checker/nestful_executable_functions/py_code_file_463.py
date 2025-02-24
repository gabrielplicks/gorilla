from typing import Any



def check_if_attribute_exists(obj: Any, attribute: str) -> bool:

    """Check if a given object's attribute exists.



    Args:

        obj: The object to check for the attribute.

        attribute: The name of the attribute to check for.



    Returns:

        A boolean value, where `True` means the attribute exists and `False` otherwise.

    """

    try:

        return hasattr(obj, attribute)  # Check if the object has the specified attribute, returning True if it does, False if not

    except Exception as e:

        print(f"Error occurred: {e}")

        return False  # Return False in case of any error

