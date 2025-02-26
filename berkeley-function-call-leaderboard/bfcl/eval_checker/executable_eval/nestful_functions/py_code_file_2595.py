from typing import List, Dict, Any



def convert_objects_to_dicts(objects: List[Any]) -> List[Dict[str, Any]]:

    """Converts a list of objects into a list of dictionaries, where each object is a dictionary keyed by attribute name and value, with the exception of attributes whose current values are None.



    Args:

        objects: A list of objects to convert.



    Returns:

        A list of dictionaries representing the objects.

    """

    dicts = []



    for obj in objects:

        obj_dict = {}

        for attr_name, attr_value in obj.__dict__.items():

            if attr_value is not None:

                obj_dict[attr_name] = attr_value

        dicts.append(obj_dict)



    return dicts

