from typing import Dict, Any



def new_object() -> Dict[Any, Any]:

    """Initializes a new empty object and returns it. If there's an error, returns None.



    Returns:

        The new empty object.

    """

    try:

        return dict()

    except Exception as e:

        return None

