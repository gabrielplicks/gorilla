from typing import List, Tuple, Any



def get_name_age_tuples(objects: List[Any]) -> List[Tuple[str, Any]]:

    """Creates a list of tuples containing the object's name and age.

    If the object does not have an age attribute, its age is replaced with None.

    Args:

        objects: A list of objects, each having a `.name` attribute.

    Returns:

        A list of tuples containing the object's name and age.

    """

    tuples = [(obj.name, obj.age if hasattr(obj, 'age') else None) for obj in objects]

    return tuples

