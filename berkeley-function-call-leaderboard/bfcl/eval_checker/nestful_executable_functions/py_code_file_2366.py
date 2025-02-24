from typing import Dict, Any



def kwargs_to_dict(kwargs: Dict[str, Any]) -> Dict[str, Any]:

    """Converts a kwargs argument to a dictionary.



    Args:

        kwargs: The kwargs argument to convert to a dictionary.



    Returns:

        A dictionary representation of the kwargs argument.

    """

    return dict(kwargs)



def get_value_from_dict(kwargs: Dict[str, Any], key: str, default: Any = None) -> Any:

    """Retrieves a value from a dictionary using the get method.



    Args:

        kwargs: The dictionary to retrieve the value from.

        key: The key to retrieve the value for.

        default: The default value to return if the key is not present.



    Returns:

        The value for the given key, or the default value if the key is not present.

    """

    return kwargs.get(key, default)

