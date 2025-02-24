from typing import List



class Object(object):

    def __init__(self, rank: int):

        self.rank = rank



def sort_objects_by_rank(objects: List[Object]) -> List[Object]:

    """Sorts a list of objects by their `.rank` attribute.



    Args:

        objects: A list of `Object` instances.



    Returns:

        A new list with the objects sorted by their `.rank` attribute.

    """

    return sorted(objects, key=lambda obj: obj.rank)

