from typing import List, TypeVar, Dict



T = TypeVar("T")



def get_class_name_and_value(objects: List[T]) -> List[Dict[str, str]]:

    """

    Returns a list of dictionaries containing the class name and value of each object in the list.

    Args:

        objects: A list of objects.

    """

    return [{obj.__class__.__name__: getattr(obj, "name")} for obj in objects]

