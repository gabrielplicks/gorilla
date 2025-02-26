from typing import Dict, List, Tuple, Any



def transform_object_list(obj_list: List[Dict[str, Any]]) -> Dict[str, Tuple[str, Any]]:

    """Transforms a list of objects into a dictionary with the object's id as the key and a tuple of (name, value) as the value.



    Args:

        obj_list: A list of objects, where each object is a dictionary with keys 'id', 'name', and 'value'.



    Returns:

        A dictionary with the object's id as the key and a tuple of (name, value) as the value.

    """

    new_dict = {}

    for obj in obj_list:

        id_value = obj['id']

        name_value = obj['name']

        value_value = obj['value']

        new_value = (name_value, value_value)

        new_dict[id_value] = new_value

    return new_dict

