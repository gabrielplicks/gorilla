from typing import List, Any



class Object:

    def __init__(self, name: str, score: float):

        self.name = name

        self.score = score



def get_highest_score_object(objects: List[Object]) -> Object:

    """

    Returns the object with the highest score from the given list of objects.

    If there is a tie, returns the first one. If the list is empty, raises a ValueError.



    Args:

        objects: A list of objects, each with a `.name` attribute and a `.score` attribute.

    """

    if len(objects) == 0:

        raise ValueError("List of objects is empty")



    highest_score_object = objects[0]

    for obj in objects[1:]:

        if obj.score > highest_score_object.score:

            highest_score_object = obj

    return highest_score_object

