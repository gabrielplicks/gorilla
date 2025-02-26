from typing import List, Dict



def list_of_objects_to_list_of_dicts(objects: List[object]) -> List[Dict[str, object]]:

    """Converts a list of objects to a list of dictionaries, each containing the object's instance variables.



    Args:

        objects: A list of objects.



    Returns:

        A list of dictionaries, each containing the object's instance variables.

    """

    dicts = []

    for obj in objects:

        vars_dict = obj.__dict__  # Get the object's instance variables

        dicts.append(vars_dict)

    return dicts

