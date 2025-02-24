import pickle

from typing import List, Tuple, Type



def serialize_objects_with_type(objects: List[object]) -> List[bytes]:

    """Serializes a list of objects using pickle and includes the type of each object.



    Args:

        objects: A list of objects to serialize.



    Returns:

        A list of serialized objects.

    """

    serialized_objects = []



    for obj in objects:

        obj_type: Type = type(obj)

        serialized_obj: bytes = pickle.dumps((obj_type, obj))

        serialized_objects.append(serialized_obj)



    return serialized_objects

