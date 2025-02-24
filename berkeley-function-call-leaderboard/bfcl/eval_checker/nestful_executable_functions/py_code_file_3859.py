from typing import List, Any



def sort_by_value(objects: List[Any]) -> List[Any]:

    """Sorts a list of objects in ascending order by their `value` attribute.

    If the `value` attribute doesn't exist on an object, then it should be sorted to the end of the list.

    Uses the `getattr` function to access the `value` attribute and handle the case where it does not exist.



    Args:

        objects: The list of objects to be sorted.



    Returns:

        The sorted list of objects.

    """

    def get_value_or_none(obj: Any) -> Any:

        try:

            return getattr(obj, 'value')

        except AttributeError:

            return None



    sorted_objects = sorted(objects, key=get_value_or_none)

    return sorted_objects

