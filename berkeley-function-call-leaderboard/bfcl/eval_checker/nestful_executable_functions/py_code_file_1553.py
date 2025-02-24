from typing import Any



def concatenate_string(string: str, obj: Any) -> str:

    """Concatenates a string with the "name" attribute of an object, if it exists.

    If the "name" attribute does not exist, the original string is returned.

    Args:

        string: The string to be concatenated.

        obj: The object to access the "name" attribute from.

    """

    name = getattr(obj, "name", None)

    if name:

        return string + name

    return string

