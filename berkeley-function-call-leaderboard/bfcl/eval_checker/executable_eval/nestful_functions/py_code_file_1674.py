from typing import Any



def is_user_defined_class(data: Any) -> str:

    """Checks if a given data type is a user-defined class and returns the class name as a string.



    Args:

        data: The data to check.



    Returns:

        The class name as a string if the data is a user-defined class, otherwise False.

    """

    if issubclass(type(data), type):

        return data.__name__

    return False

