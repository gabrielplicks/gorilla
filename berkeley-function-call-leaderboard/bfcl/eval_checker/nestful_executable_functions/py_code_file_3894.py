from typing import List, Dict



def find_coarsest(objects: List[Dict[str, float]]) -> Dict[str, float]:

    """Finds the coarsest object in a list of objects.



    The coarsest object is the one with the lowest value for the key 'v'.



    Args:

        objects: A list of objects, where each object is a dictionary with a key 'v'.



    Returns:

        The coarsest object. If the list is empty, returns None.

    """

    if len(objects) == 0:

        return None



    coarsest = objects[0]

    for obj in objects[1:]:

        if obj['v'] < coarsest['v']:

            coarsest = obj



    return coarsest

