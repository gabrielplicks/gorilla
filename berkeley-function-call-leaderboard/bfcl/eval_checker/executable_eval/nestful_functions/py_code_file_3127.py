from typing import List, Any



class Object:

    def __init__(self, id: int, name: str):

        self.id = id

        self.name = name



def get_object_name_by_id(objects: List[Object], id: int) -> str:

    """Finds the object with the given id and returns its name.

    Args:

        objects: A list of objects.

        id: The id of the object to find.

    """

    for obj in objects:

        if obj.id == id:

            return obj.name

    return f"Object not found for id: {id}"

