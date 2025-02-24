from typing import Any



def retrieve_updated_at(data: Any) -> str:

    """Retrieves the "updated_at" value from a data structure that allows attribute access.

    If the "updated_at" attribute does not exist, defaults to a static value of "1970-01-01T00:00:00Z".



    Args:

        data: The data structure that supports attribute access.



    Returns:

        The "updated_at" value or the default value if it does not exist.

    """

    try:

        return data.updated_at

    except AttributeError:

        return "1970-01-01T00:00:00Z"

