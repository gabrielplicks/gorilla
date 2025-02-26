from typing import Dict, List, Any



class Object:

    def __init__(self, id: int):

        self.id = id



def map_objects_by_id(objects: List[Object]) -> Dict[int, Any]:

    """Maps each object's `id` to the object itself.



    Args:

        objects: A list of objects, each containing an `id` attribute.



    Returns:

        A dictionary mapping each object's `id` to the object itself.

    """

    return {obj.id: obj for obj in objects}

