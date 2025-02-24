from typing import Any



def hello_message(obj: Any) -> str:

    """Returns a string message formatted as `Hello, <name>` if the object has an attribute named `name`,

    or `Hello, world` if the object does not have such an attribute.

    If the object is not a class instance, returns `Hello, world` instead.



    Args:

        obj: The object to check for the `name` attribute.

    """

    if isinstance(obj, object):

        if hasattr(obj, 'name'):

            return f"Hello, {obj.name}"

        else:

            return "Hello, world"

    else:

        return "Hello, world"

