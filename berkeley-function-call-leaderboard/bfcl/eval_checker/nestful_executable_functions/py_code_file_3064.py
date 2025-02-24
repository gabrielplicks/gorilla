from typing import Any



def check_and_concatenate(obj: Any, attribute_name: str, string: str) -> str:

    """Checks if a given object has an attribute with a given name, concatenates the attribute value to a string, and then returns the modified string. If the attribute does not exist, returns the original string without any changes.



    Args:

        obj: The object to check for the attribute.

        attribute_name: The name of the attribute to check.

        string: The string to concatenate the attribute value to.

    """

    if hasattr(obj, attribute_name):

        attr_value = getattr(obj, attribute_name)

        return string + str(attr_value)

    else:

        return string

