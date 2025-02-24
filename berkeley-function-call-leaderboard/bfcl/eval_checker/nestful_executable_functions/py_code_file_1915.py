from typing import Any



def create_id(obj: Any, prefix: str = None) -> str:

    """Generates an ID for the given object by combining the object's type name and the object's value.

    If no `prefix` argument is provided, it should just return the object's type name.

    Args:

        obj: The object to generate an ID for.

        prefix: The prefix to add to the ID.

    """

    type_name = type(obj).__name__

    value = str(obj)

    id = type_name + "_" + value

    if prefix is not None:

        id = prefix + "_" + id

    return id

