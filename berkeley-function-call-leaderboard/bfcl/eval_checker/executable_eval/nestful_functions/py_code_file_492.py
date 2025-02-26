from typing import Dict, List, Type



def group_objects_by_type(objects: List) -> Dict[Type, List]:

    """Groups objects by their types.



    Args:

        objects: A list of objects.



    Returns:

        A dictionary with each object's type as a key and a list of objects of that type as a value.

    """

    grouped_objects = {}



    try:

        for obj in objects:

            obj_type = type(obj)

            if obj_type not in grouped_objects:

                grouped_objects[obj_type] = []

            grouped_objects[obj_type].append(obj)

    except Exception as e:

        print(f"Error occurred: {e}")



    return grouped_objects

