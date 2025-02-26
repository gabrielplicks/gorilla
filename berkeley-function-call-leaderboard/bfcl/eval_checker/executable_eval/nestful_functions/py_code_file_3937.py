from typing import Any



def get_object_type(name: str) -> str:

    """

    Returns the type of the object by the given name.

    The object is assumed to be in the global scope and may be a module, class, function, method, built-in function, or any other type of object.

    If the object does not exist, a `NameError` exception is handled and the function returns "object not found".



    Args:

        name: The name of the object.



    Returns:

        The type of the object or "object not found" if the object does not exist.

    """

    try:

        obj = eval(name)

        return type(obj).__name__

    except NameError:

        return "object not found"

