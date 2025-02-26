from typing import List



class Object:

    def __init__(self, is_active: bool):

        self.is_active = is_active



def get_active_objects(objects: List[Object]) -> List[Object]:

    """Returns a list of the objects with the attribute `is_active` set to `True`.



    Args:

        objects: A list of objects.



    Returns:

        A list of objects with `is_active` set to `True`.

    """

    return [obj for obj in objects if obj.is_active]

