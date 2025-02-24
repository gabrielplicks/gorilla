from typing import Any, Dict, List



def update_object_attributes(obj: Any, pattern: List[Dict[str, Any]]) -> None:

    """Updates an object's attributes according to a specific pattern.



    Args:

        obj: The object to update.

        pattern: A list of dictionaries containing attribute names and their new values.

    """

    for attr_dict in pattern:

        attr_name, attr_value = list(attr_dict.items())[0]

        if not hasattr(obj, attr_name):

            setattr(obj, attr_name, attr_value)

        else:

            setattr(obj, attr_name, attr_value)

