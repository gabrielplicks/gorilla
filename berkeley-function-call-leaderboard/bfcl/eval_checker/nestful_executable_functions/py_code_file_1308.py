from typing import Any, List



def get_attributes_with_string(obj: Any, target_string: str) -> List[str]:

    """Returns a list of all the attributes of the given object whose names contain the given string.



    Args:

        obj: The object to search for attributes.

        target_string: The string to search for in the attribute names.

    """

    attributes = dir(obj)

    filtered_attributes = list(filter(lambda a: target_string in a, attributes))

    return filtered_attributes

