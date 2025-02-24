from typing import Any



def check_identity(obj1: Any, obj2: Any) -> bool:

    """Checks if two objects are considered identical, where object identity is defined as having the same memory address.

    Args:

        obj1: The first object to compare.

        obj2: The second object to compare.

    """

    return id(obj1) == id(obj2)

