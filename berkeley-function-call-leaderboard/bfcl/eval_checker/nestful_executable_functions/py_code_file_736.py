from typing import Dict, List, Any



def collect_values(objects: List[Dict[str, Any]]) -> Dict[str, List[Any]]:

    """Collects values of properties from a list of objects into a dictionary.



    Each property is a key in the dictionary, and the corresponding value is a list of those property values for all the objects in the list.



    Args:

        objects: A list of objects, where each object is a dictionary with string keys and any values.



    Returns:

        A dictionary where each key is a property and the corresponding value is a list of those property values for all the objects in the list.

    """

    return {prop: [obj[prop] for obj in objects] for prop in objects[0]}

