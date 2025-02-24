from typing import List, Any



def sort_objects_by_name_or_title(objects: List[Any]) -> List[Any]:

    """Sorts a list of objects by their name or title attribute in ascending order.



    Args:

        objects: A list of objects to be sorted.



    Raises:

        AttributeError: If an object does not have a name or title attribute.

    """

    def get_name_or_title(obj: Any) -> str:

        try:

            return getattr(obj, "name")

        except AttributeError:

            try:

                return getattr(obj, "title")

            except AttributeError:

                raise AttributeError("Object does not have a name or title attribute")



    return sorted(objects, key=get_name_or_title)

