from typing import Any



def get_last_name(obj: Any) -> str:

    """Returns the `last_name` attribute of an object if it is not null, otherwise returns `None`.



    Args:

        obj: The object to get the `last_name` attribute from.

    """

    if obj is not None:

        return obj.last_name

    else:

        return None

