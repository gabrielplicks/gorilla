from typing import List, Any



def find_by_attribute(objects: List[Any], attribute: str, value: Any) -> List[Any]:

    """Finds objects with a specified attribute and value.



    Args:

        objects: A list of objects.

        attribute: The attribute to search for.

        value: The value to match.



    Returns:

        A list of objects with the specified attribute and value.

    """

    return list(filter(lambda obj: getattr(obj, attribute) == value, objects))

