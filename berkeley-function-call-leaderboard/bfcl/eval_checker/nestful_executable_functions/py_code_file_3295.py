from typing import Any



def has_private_attributes(obj: Any) -> bool:

    """Checks if an object has any private attributes.



    Args:

        obj: The object to check.



    Returns:

        True if the object has any private attributes, False otherwise.

    """

    for attr in dir(obj):

        if attr.startswith('__'):

            return True



    return False

