from typing import List, Any



def compute_average_value(objects: List[Any], attribute: str) -> float:

    """Computes the average value of a given attribute of a list of objects.



    Args:

        objects: A list of objects.

        attribute: The name of the attribute to compute the average value of.



    Returns:

        The average value of the specified attribute, or None if no objects are provided or all the objects don't have the specified attribute.

    """

    if not objects:

        return None



    total = 0

    num_objects = 0



    for obj in objects:

        if hasattr(obj, attribute):

            total += getattr(obj, attribute)

            num_objects += 1



    if num_objects == 0:

        return None



    return total / num_objects

