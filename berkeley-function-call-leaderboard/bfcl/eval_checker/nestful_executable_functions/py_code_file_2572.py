from typing import List, Any



def sort_by_attribute(objects: List[Any], attr: str, reverse: bool = False) -> List[Any]:

    """Sorts a list of objects by a given attribute name.



    Args:

        objects: The list of objects to be sorted.

        attr: The name of the attribute to sort by.

        reverse: A boolean indicating whether to sort in descending order.

    """

    return sorted(objects, key=lambda obj: getattr(obj, attr), reverse=reverse)

