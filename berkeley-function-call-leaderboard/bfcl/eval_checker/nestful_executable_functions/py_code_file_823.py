from typing import Dict, Any



def get_arg(args: Dict[str, Any], key: str) -> str:

    """Retrieves the value of an element from an argument dictionary based on its key.

    Args:

        args: A dictionary containing the arguments.

        key: The key of the element to retrieve.

    Returns:

        The value of the element if it is a non-empty string, otherwise None.

    """

    if key in args:

        val = args[key]

        if isinstance(val, str) and len(val) > 0:

            return val

    return None

