from typing import List, Set



def get_flattened_ids(objects: List[object]) -> List[int]:

    """Returns a list of unique IDs from a list of objects, where each object has a `related_ids` attribute.



    Args:

        objects: A list of objects, where each object has a `related_ids` attribute.



    Returns:

        A list of unique IDs, sorted in ascending order.

    """

    return sorted(set(related_id for obj in objects for related_id in obj.related_ids))

