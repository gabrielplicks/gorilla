from typing import Dict



def string_interpolation(kwargs: Dict[str, str]) -> str:

    """

    Returns a formatted string using f-string interpolation with the keys in kwargs as field names.

    Args:

        kwargs: A dictionary containing the field names and their corresponding values.

    """

    if not kwargs:

        return ""

    formatted_string = f"Hello, {kwargs['name']}!"



    return formatted_string

