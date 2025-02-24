from typing import Dict



def dict_to_string_literal(d: Dict[str, str]) -> str:

    """Creates a string literal representation of a dictionary.



    The dictionary's keys are enclosed in double quotes and the values are enclosed in single quotes.



    Args:

        d: The dictionary to convert to a string literal.



    Returns:

        A string literal representation of the dictionary.

    """

    string_literals = []

    for key, value in d.items():

        string_literals.append(f"'{key}': {value!r}")

    return "{" + ", ".join(string_literals) + "}"

