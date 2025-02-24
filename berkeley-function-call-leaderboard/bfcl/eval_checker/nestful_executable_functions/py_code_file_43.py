from typing import Union



def convert_boolean_to_string(boolean_value: Union[bool, int]) -> str:

    """Converts a boolean value to a string "true" or "false".

    Args:

        boolean_value: The boolean value to be converted.

    """

    if boolean_value:

        return "true"

    else:

        return "false"

