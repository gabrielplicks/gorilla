from typing import List



def find_unique_objects(objects: List[object]) -> List[object]:

    """Finds unique objects from a given list.



    Args:

        objects: A list of objects.



    Returns:

        A list of unique objects.

    """

    seen = set()

    unique_objects = []

    for obj in objects:

        if obj not in seen:

            seen.add(obj)

            unique_objects.append(obj)

    return unique_objects

