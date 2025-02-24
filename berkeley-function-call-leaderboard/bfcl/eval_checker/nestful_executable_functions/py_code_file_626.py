from typing import List, Dict



def find_parent_key(objects: List[Dict], target_key: str) -> str:

    """Finds the parent key of a target object in a list of objects.

    Args:

        objects: A list of dictionaries representing a set of objects.

        target_key: The key of the target object.

    Returns:

        The parent key of the target object, if a parent relationship exists.

        None if there is no parent or if the target object is not found in the list.

    """

    for obj in objects:

        if obj["key"] == target_key:

            if "parentKey" in obj:

                return obj["parentKey"]

    return None

