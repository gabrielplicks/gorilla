from typing import List, Dict, Any



def map_children_to_id(objects: List[Dict[str, Any]]) -> Dict[str, List[str]]:

    """Maps each object's "id" key to a list of its "children" objects.



    Args:

        objects: A list of objects, where each object is a dictionary with "id" and "children" keys.



    Returns:

        A dictionary that maps each object's "id" key to a list of its "children" objects.

    """

    return {obj["id"]: [child["id"] for child in obj["children"]] if "children" in obj else [] for obj in objects}

