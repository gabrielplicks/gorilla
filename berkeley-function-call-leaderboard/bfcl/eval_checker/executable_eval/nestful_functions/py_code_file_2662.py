from typing import Dict, Any



def retrieve_names(obj_dict: Dict[str, Any]) -> list:

    """Retrieves the values of the names attribute of all objects in a dictionary that have the names attribute.



    Args:

        obj_dict: A dictionary where the values are objects that may have a names attribute.



    Returns:

        A list of names.

    """

    result = []

    for key in obj_dict:

        if hasattr(obj_dict[key], 'names'):

            result.append(obj_dict[key].names)

    return result

