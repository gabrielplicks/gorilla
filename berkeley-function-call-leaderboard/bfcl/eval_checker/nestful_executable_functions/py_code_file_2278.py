from typing import Dict, List, Any



class Object:

    def __init__(self, id: int, name: str):

        self.id = id

        self.name = name



def create_id_to_name_mapping(objects: List[Object]) -> Dict[int, str]:

    """Creates a dictionary mapping `id` to `name` for a list of objects.



    Args:

        objects: A list of objects with `id` and `name` attributes.



    Returns:

        A dictionary mapping `id` to `name`.

    """

    mapping = {}

    for obj in objects:

        mapping[obj.id] = obj.name

    return mapping

