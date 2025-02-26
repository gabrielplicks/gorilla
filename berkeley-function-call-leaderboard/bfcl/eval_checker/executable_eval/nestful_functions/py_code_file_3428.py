from typing import Dict



def username_extractor(input_dict: Dict[str, str]) -> tuple:

    """Extracts the `username` key, deletes the `password` key, and returns the `is_admin` value.

    If the input dictionary does not have a `username` key, set the value of `username` to `None`.

    Args:

        input_dict: A dictionary object.

    Returns:

        A tuple containing the `username` and `is_admin` values.

    """

    username = input_dict.get("username", None)

    input_dict.pop("password", None)

    is_admin = input_dict.get("is_admin", False)

    return username, is_admin

