from typing import Any, Union



def check_object_type(obj: Any) -> Union[int, None]:

    """Checks if a given object is a string, list, tuple, or dictionary.

    If it is any of these types, returns the object's length.

    If it is none of these types, returns `None`.



    Args:

        obj: The object to check.

    """

    if isinstance(obj, str):

        return len(obj)

    elif isinstance(obj, list):

        return len(obj)

    elif isinstance(obj, tuple):

        return len(obj)

    elif isinstance(obj, dict):

        return len(obj)

    else:

        return None

