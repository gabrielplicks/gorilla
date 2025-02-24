import json

from typing import Any, Dict, List



def serialize_nested_object(obj: Any) -> Dict[str, Any]:

    """Serializes a complex object with a nested structure of lists, dictionaries, and strings into a JSON representation.

    The JSON representation includes the keys of the topmost dictionary and all nested lists and dictionaries.

    The names "list" and "dict" are used for lists and dictionaries, respectively, in the JSON representation.



    Args:

        obj: The complex object to serialize.



    Returns:

        A dictionary representing the serialized object.

    """

    if isinstance(obj, dict):

        serialized_dict = {"dict": {key: serialize_nested_object(value) for key, value in obj.items()}}



        return serialized_dict

    elif isinstance(obj, list):

        return {"list": [serialize_nested_object(item) for item in obj]}

    else:

        return obj

