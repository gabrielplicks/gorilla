from typing import Any, List



def get_integer_attributes(obj: Any) -> List[str]:

    """Returns a list of all attribute names which have an integer value.



    Args:

        obj: The object to inspect.



    Returns:

        A list of attribute names which have an integer value.

    """

    result = []

    for attr in dir(obj):

        if isinstance(getattr(obj, attr), int):

            result.append(attr)

    return result

