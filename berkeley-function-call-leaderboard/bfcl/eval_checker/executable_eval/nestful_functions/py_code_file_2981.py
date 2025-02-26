from typing import Any



def parse_integer_attribute(obj: Any, attr_name: str, default_value: int = 0) -> int:

    """Parses the integer value of the given attribute from the PyObject object.



    Args:

        obj: The PyObject object.

        attr_name: The name of the attribute to parse.

        default_value: The default integer value to return if the attribute is not found.

    """

    if hasattr(obj, attr_name):

        return int(getattr(obj, attr_name))

    return default_value

