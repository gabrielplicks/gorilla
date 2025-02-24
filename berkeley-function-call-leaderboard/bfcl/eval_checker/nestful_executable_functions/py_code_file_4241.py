from collections import defaultdict

from typing import Dict, List, Union



def group_by_id(objects: List[Dict[str, Union[str, int]]]) -> Dict[Union[str, int], List[Dict[str, Union[str, int]]]]:

    """Groups a list of objects by their 'id' property.



    Args:

        objects: A list of objects, each expected to have an 'id' property.



    Returns:

        A dictionary where the keys are the 'id' properties of the objects and the values are lists of objects with the same 'id' property.

    """

    grouped = defaultdict(list)

    for obj in objects:

        if 'id' in obj:

            grouped[obj['id']].append(obj)

    return grouped

