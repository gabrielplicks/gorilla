from typing import Dict



def string_from_dict(dictionary: Dict[str, str]) -> str:

    """Constructs a string from a dictionary of string keys and values.

    Each key-value pair is represented as key=value, separated by commas and enclosed in curly braces.

    If a key has no corresponding value, it is represented as key=.



    Args:

        dictionary: A dictionary of string keys and values.



    Returns:

        A string representation of the dictionary in the desired format.

    """

    return "{" + ", ".join([f"'{k}'='{v}'" if v is not None else f"'{k}'=" for k, v in dictionary.items()]) + "}"

