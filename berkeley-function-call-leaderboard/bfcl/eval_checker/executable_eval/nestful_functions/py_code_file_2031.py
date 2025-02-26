from typing import Any



def get_info_with_error_handling(key: str, data: dict[str, Any]) -> str:

    """

    Returns a string with the key and value separated by a colon and a space.

    Raises a KeyError if the key is not in the data dictionary.

    Raises a ValueError if the value is not a string.



    Args:

        key: The key to look up in the data dictionary.

        data: The data dictionary.



    Returns:

        A string with the key and value separated by a colon and a space.

    """

    if key not in data:

        raise KeyError('Key not found in the data dictionary.')



    value = data[key]

    if not isinstance(value, str):

        raise ValueError('Value is not a string.')



    return f'{key}: {value}'

