from typing import List, Any



def search_objects(obj_list: List[Any], attr_name: str) -> List[Any]:

    """Searches through a list of objects and returns a subset of those objects that have a certain attribute.

    Args:

        obj_list: The list of objects to search through.

        attr_name: The name of the attribute to search for.

    """

    result = []

    for obj in obj_list:

        if hasattr(obj, attr_name):

            result.append(obj)

        else:

            print(f"Attribute '{attr_name}' not found on object '{obj}'")

    return result

