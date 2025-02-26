from typing import Any, Dict



def construct_object_dict(obj: Any) -> Dict[str, Any]:

    """Constructs a dictionary containing the object's public properties and methods.



    Args:

        obj: The object reference.



    Returns:

        A dictionary containing the object's public properties and methods.

    """

    obj_dict = {}

    for attr in dir(obj):

        if not attr.startswith('_'):

            obj_dict[attr] = getattr(obj, attr)

    return obj_dict

