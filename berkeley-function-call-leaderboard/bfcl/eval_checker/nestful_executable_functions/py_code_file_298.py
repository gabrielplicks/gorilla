from typing import List, Dict



def filter_objects_with_depth(objects: List[Dict]) -> List[Dict]:

    """Filters a list of objects and returns a list of objects with a positive depth attribute.



    Args:

        objects: A list of objects.



    Returns:

        A list of objects with a positive depth attribute.

    """

    return list(filter(lambda obj: obj.get("depth") is not None and obj.get("depth") > 0, objects))

