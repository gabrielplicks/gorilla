from typing import Any



def check_same_class(obj1: Any, obj2: Any) -> bool:

    """Checks whether two objects are instances of the same class, including checking for inheritance.



    Args:

        obj1: The first object to be checked.

        obj2: The second object to be checked.



    Returns:

        True if the objects are instances of the same class or if one is a subclass of the other, False otherwise.

    """

    if isinstance(obj1, type(obj2)):

        return True

    if issubclass(type(obj1), type(obj2)):

        return True

    return False

