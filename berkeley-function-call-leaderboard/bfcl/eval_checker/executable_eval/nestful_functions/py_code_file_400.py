from typing import List, Any



def get_average_length_of_string_attribute(objects: List[Any], attribute_name: str) -> float:

    """Calculates the average length of a specified string attribute across all objects.

    Args:

        objects: A list of objects.

        attribute_name: The name of the string attribute.

    """

    if not objects or not attribute_name:

        return None



    total_length = 0

    num_objects = 0



    for obj in objects:

        if hasattr(obj, attribute_name) and isinstance(getattr(obj, attribute_name), str):

            total_length += len(getattr(obj, attribute_name))

            num_objects += 1



    if num_objects > 0:

        return total_length / num_objects

    else:

        return None

