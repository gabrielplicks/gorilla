from typing import List, Any



def get_attr_str(objects: List[Any], attr: str) -> str:

    """Returns a string representation of the specified attribute values for a list of objects.

    Args:

        objects: A list of objects.

        attr: The name of the attribute to retrieve.

    """

    attr_strs = []

    for obj in objects:

        attr_value = getattr(obj, attr, None)

        attr_str = str(attr_value) if attr_value is not None else "None"

        attr_strs.append(attr_str)

    return " ".join(attr_strs)

