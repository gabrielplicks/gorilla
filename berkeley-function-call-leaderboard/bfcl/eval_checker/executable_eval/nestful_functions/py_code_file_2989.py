from typing import Any



def format_obj(obj: Any) -> str:

    """Formats an object into a string by concatenating its `id` and `name` attributes, separated by a colon.



    Args:

        obj: The object to be formatted.

    """

    obj_id = obj.id

    obj_name = obj.name

    result = str(obj_id) + ":" + obj_name

    return result

