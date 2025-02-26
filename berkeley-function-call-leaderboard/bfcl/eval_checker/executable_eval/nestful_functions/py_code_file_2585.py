from typing import List, Any



def count_object_attributes(objects: List[Any], attribute: str) -> int:

    """

    Counts the number of objects in `objects` that have an attribute named `attribute`.



    Args:

        objects: A list of objects to check for the attribute.

        attribute: The name of the attribute to check for.



    Returns:

        The number of objects that have the attribute.

    """

    return len([obj for obj in objects if hasattr(obj, attribute)])

