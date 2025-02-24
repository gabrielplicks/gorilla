from typing import Dict



def bool_to_str(value: bool) -> str:

    """Converts a boolean value to its string representation.



    Args:

        value: The boolean value to convert.



    Returns:

        The string representation of the input value.

    """

    mapping: Dict[bool, str] = {True: "True", False: "False"}

    return mapping[value]

