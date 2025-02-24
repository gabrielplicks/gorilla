from typing import Any



def is_not_str_or_bytes(value: Any) -> bool:

    """Checks if a given value is not a string or bytes object.



    Args:

        value: The value to check.



    Returns:

        True if the value is not a string or bytes object, False otherwise.

    """

    return not (isinstance(value, str) or isinstance(value, bytes))

