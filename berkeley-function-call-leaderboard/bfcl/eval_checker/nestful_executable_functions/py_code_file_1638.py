from typing import Any



def is_valid_property_value(obj: Any, property_name: str) -> bool:

    """Checks if the input object has a property with a value of type int or float and the value is in the range of -100 to 100.



    Args:

        obj: The input object.

        property_name: The name of the property to check.



    Returns:

        True if the property exists, its value is of type int or float, and the value is in the range of -100 to 100. Otherwise, False.

    """

    if not hasattr(obj, property_name):

        return False

    property_value = getattr(obj, property_name)

    if not isinstance(property_value, (int, float)):

        return False

    if -100 <= property_value <= 100:

        return True

    else:

        return False

