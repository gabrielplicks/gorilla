from typing import Any, List



def sort_list_by_attr(obj_list: List[Any], attr_name: str) -> None:

    """Sorts a list of objects in-place by the value of a given attribute.



    If the attribute is not available for an object, the object remains in its original position.

    Raises an exception if the given attribute does not exist for any of the objects in the list.



    Args:

        obj_list: A list of objects to be sorted.

        attr_name: The name of the attribute to sort by.

    """

    if not all(hasattr(obj, attr_name) for obj in obj_list):

        raise AttributeError("Attribute not found in all objects")

    obj_list.sort(key=lambda obj: getattr(obj, attr_name))

