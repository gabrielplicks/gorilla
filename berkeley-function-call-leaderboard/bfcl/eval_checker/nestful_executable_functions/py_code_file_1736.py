from typing import List, Dict



def get_object_names(objects: List[Dict], attribute_name: str) -> List[str]:

    """Returns a list of strings with the value of the specified attribute appended to each object's name, separated by a comma and a space.



    Args:

        objects: A list of objects, each object has an attribute called `name`.

        attribute_name: A string representing a valid attribute of each object.



    Returns:

        A list of strings.

    """

    return [f"{obj['name']}, {obj[attribute_name]}" for obj in objects]

