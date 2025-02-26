from typing import List, Dict



def reshape_objects(objects: List[Dict]) -> List[List[Dict]]:

    """

    Reshapes a list of objects with the same attributes into a nested list of lists,

    where each inner list contains only one object. The result should have the same

    attribute names as the original list, and the values should be in the same order

    as the original list.

    Args:

        objects: A list of dictionaries, where each dictionary represents an object

                  with the same attributes.

    Returns:

        A nested list of lists, where each inner list contains only one object.

    """

    reshaped = []



    for obj in objects:

        reshaped.append([obj])



    return reshaped

