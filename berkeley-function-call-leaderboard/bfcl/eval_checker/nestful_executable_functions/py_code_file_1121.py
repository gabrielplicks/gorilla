from typing import Any



class CustomException(Exception):

    """A custom exception class."""

    pass



def check_has_attr(obj: Any, attr_name: str) -> None:

    """Check if the provided object has a certain attribute.



    If the object does not have the attribute, raise a custom exception containing the attribute name and a message that says "This object does not have a {attribute_name} attribute".



    Args:

        obj: The object to check.

        attr_name: The name of the attribute to check for.



    Raises:

        CustomException: If the object does not have the specified attribute.

    """

    if not hasattr(obj, attr_name):

        raise CustomException(f"This object does not have a {attr_name} attribute")

