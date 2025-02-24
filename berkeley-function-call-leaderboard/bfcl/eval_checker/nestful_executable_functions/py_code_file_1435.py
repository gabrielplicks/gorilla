from typing import Any



def find_documented_method(obj: Any) -> str:

    """Finds the first documented method in the class hierarchy of a given object.



    Args:

        obj: The object to search for a documented method.



    Returns:

        The name of the first documented method found in the class hierarchy.

    """

    obj_class = obj.__class__

    if hasattr(obj_class, "__doc__") and obj_class.__doc__ is not None:

        for method in obj_class.__dict__.values():

            if callable(method) and method.__doc__ is not None:

                return method.__name__

    return find_documented_method(obj_class)

