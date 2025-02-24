from typing import List, Tuple



class Object:

    def __init__(self, name: str):

        self.name = name



def filter_by_name_and_type(objects: List[Object], filter_criteria: List[Tuple[str, bool]]) -> List[Object]:

    """

    Filters a list of objects based on their names and a list of filter criteria.



    Args:

        objects: A list of objects to filter.

        filter_criteria: A list of tuples containing a string and a boolean value.



    Returns:

        A filtered list of objects that match all filter criteria.

    """

    filtered_objects = []

    for obj in objects:

        for name, include in filter_criteria:

            if obj.name == name and include:

                filtered_objects.append(obj)

    return filtered_objects

