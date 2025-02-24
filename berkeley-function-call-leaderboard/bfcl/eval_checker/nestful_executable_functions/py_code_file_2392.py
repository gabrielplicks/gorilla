from typing import Any



def has_foo_attribute(obj: Any) -> bool:

    """Checks whether an object has an attribute named `foo` and returns `True` if it does, `False` otherwise.



    Args:

        obj: The object to check for the attribute.



    Returns:

        `True` if the object has an attribute named `foo`, `False` otherwise.

    """

    try:

        getattr(obj, "foo")

        return True

    except AttributeError:

        return False

